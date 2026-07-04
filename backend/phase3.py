from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import time
import base64
import os
from pydantic import BaseModel, Field
from phase2_planner import research
from dotenv import load_dotenv
import uuid

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)


# ---------------------------------------------------------------------------
# Schemas — image / design
# ---------------------------------------------------------------------------
class SectionBBox(BaseModel):
    x: float = Field(description="Left edge, as % of image width (0-100)")
    y: float = Field(description="Top edge, as % of image height (0-100)")
    width: float = Field(description="Width, as % of image width")
    height: float = Field(description="Height, as % of image height")


class ObservedColor(BaseModel):
    hex_estimate: str = Field(description="Best-guess hex code for this color, e.g. '#1a1a2e'")
    usage: str = Field(description="Where this color is used, e.g. 'button background', 'page background', 'heading text'")


class Typography(BaseModel):
    role: str = Field(description="e.g. 'h1', 'h2', 'body', 'caption', 'button label'")
    relative_size: str = Field(description="largest | large | medium | small | smallest — relative to other text on the page")
    weight: str = Field(description="bold | semibold | regular | light")
    style_notes: str = Field(description="uppercase, italic, letter-spacing look, underline, etc. — only if visible")


class Component(BaseModel):
    component_type: str = Field(description="button | input | card | image | icon | heading | paragraph | link | badge | avatar | divider | nav_item | logo | other")
    text_content: str = Field(description="Exact visible text, character for character. Empty string if none.")
    bbox: SectionBBox = Field(description="Bounding box as % of the FULL image (not section-relative)")
    colors: List[ObservedColor] = Field(description="Colors specific to this component (background, text, border)")
    typography: Typography = Field(description="Fill only if this component contains text")
    image_description: str = Field(description="If component_type is image/icon/avatar: describe what it visually depicts. Empty otherwise.")
    notes: str = Field(description="Rounded corners, shadow, border, spacing around it — anything a developer needs to recreate it")


class PageSection(BaseModel):
    section_id: str = Field(description="Short unique id, e.g. 'nav', 'hero', 'footer'")
    section_type: str = Field(description="nav | hero | feature_grid | testimonial | pricing | footer | sidebar | form | content | other")
    bbox: SectionBBox
    layout_mode: str = Field(description="flex-row | flex-column | grid | stacked | overlay")
    column_count: int
    background: ObservedColor = Field(description="Section's background color/treatment")
    spacing_density: str = Field(description="tight | normal | spacious")
    alignment: str = Field(description="left | center | right | justified")
    components: List[Component] = Field(description="All components in this section, in reading order")


class PageSpec(BaseModel):
    overall_layout: str = Field(description="single-column | sidebar-main | dashboard-grid | landing-sections | other")
    page_background: ObservedColor
    sections: List[PageSection] = Field(description="Top-to-bottom ordered list of every section on the page")


class research_need(BaseModel):
    research_needed: Literal["true", "false"] = Field(description="whether the research is needed or not")


# ---------------------------------------------------------------------------
# Schemas — file plan (FIXED: now matches everything the planner prompt promises)
# ---------------------------------------------------------------------------
class fileplan(BaseModel):
    filename: str = Field(description="exact filename with extension")
    purpose: str = Field(description="one clear sentence on what this file does")
    responsibilities: List[str] = Field(
        description="Specific things this file handles and nothing else — including exact "
                    "text, colors, and layout details from design_schema where relevant"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Filenames this file imports, links to, or requires (empty list if none)",
    )
    generate_order: int = Field(description="Integer starting from 1. Lower number = generate first.")
    language: str = Field(description="Language this file should be written in")
    package: str = Field(default="", description="Package needed for this file, if any (empty string if none)")


class all_files(BaseModel):
    project_type: str = Field(description="e.g. 'static website', 'REST API', 'CLI tool'")
    language: str = Field(description="Primary language/stack chosen for the project")
    reasoning: str = Field(description="One or two sentences explaining why this stack was chosen")
    used_research: bool = Field(description="Whether research_report informed this plan")
    used_design_schema: bool = Field(description="Whether design_schema informed this plan")
    packages: List[str] = Field(default_factory=list, description="Packages needed for the whole project")
    files: List[fileplan]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class State(TypedDict):
    prompt: str
    reference_image_path: str
    new_request: str
    overall_image_design: PageSpec | None
    image_description: str
    research_needed: Literal["true", "false"]
    research_context: str
    fileplans: all_files
    file_code: dict[str, str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def query_optimizer(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are an expert software engineer and project planner.
        Your ONLY job is to take the Raw User Request below and turn it into a clear, concise, and optimized engineering prompt.
        Do NOT write an introduction. Do NOT say "Okay, I understand". Output ONLY the clean, optimized prompt text.

        ---
        RAW USER REQUEST:
        {query}
        ---

        OPTIMIZED ENGINEERING PROMPT""",
        input_variables=['query']
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({'query': state['prompt']})
    print("Finished query optimizer")
    return {'new_request': response}


def get_image_schema(state: State) -> State:
    """FIXED: guards against missing/empty path, never leaves a variable unbound,
    always returns a proper state dict."""
    path = state.get('reference_image_path', '')

    if not path or not os.path.exists(path):
        return {'overall_image_design': None}

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"

    message = HumanMessage(content=[
        {"type": "text", "text": "Give me the complete detail of this image — every section, component, text, color, and layout detail, following the schema."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{image_b64}"}
    ])

    structured_model = model.with_structured_output(PageSpec)
    result = structured_model.invoke([message])
    print("Finished image analysis")
    return {'overall_image_design': result}


def format_design_schema(state: State) -> State:
    """FIXED: always returns a dict (never a bare string) so downstream
    state['image_description'] lookups never KeyError."""
    schema = state.get('overall_image_design')
    if not schema:
        return {'image_description': "No reference image was provided."}

    lines = [
        f"Overall layout: {schema.overall_layout}",
        f"Page background: {schema.page_background.hex_estimate} ({schema.page_background.usage})",
        ""
    ]
    for section in schema.sections:
        lines.append(f"SECTION [{section.section_id}] - type: {section.section_type}")
        lines.append(f"  Position: x={section.bbox.x}%, y={section.bbox.y}%, w={section.bbox.width}%, h={section.bbox.height}%")
        lines.append(f"  Layout: {section.layout_mode}, columns: {section.column_count}, alignment: {section.alignment}, spacing: {section.spacing_density}")
        lines.append(f"  Background: {section.background.hex_estimate} ({section.background.usage})")
        for comp in section.components:
            lines.append(f"  - COMPONENT [{comp.component_type}] text: \"{comp.text_content}\"")
            lines.append(f"    Position: x={comp.bbox.x}%, y={comp.bbox.y}%, w={comp.bbox.width}%, h={comp.bbox.height}%")
            for c in comp.colors:
                lines.append(f"    Color: {c.hex_estimate} ({c.usage})")
            if comp.text_content:
                t = comp.typography
                lines.append(f"    Typography: {t.role}, size={t.relative_size}, weight={t.weight}, notes={t.style_notes}")
            if comp.image_description:
                lines.append(f"    Image depicts: {comp.image_description}")
            if comp.notes:
                lines.append(f"    Notes: {comp.notes}")
        lines.append("")

    return {'image_description': "\n".join(lines)}


def call_researchor_or_not(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are an expert software engineer and project planner for an AI code generation system.
Your ONLY job is to analyze the user's request and determine if external research is needed before planning the project.
You do NOT write any code. You only decide if research is necessary.
---
USER REQUEST TO EVALUATE:
{query}
---
YOUR TASK:
Determine whether this request requires research about an external, real-world site, product, brand, or reference
before it can be planned and built accurately.

Set research_needed = "true" if the request:
- Explicitly asks to "clone", "replicate", "copy", or build something "like" or "inspired by" a specific named
  real product, company, or website (e.g. "inspired by Vercel's dashboard", "clone Stripe's homepage").
- References specific real-world UI/UX patterns, layouts, or branding that would require knowing what that
  actual site looks like to be built accurately.
- Mentions a specific real API, library, or service whose current behavior/structure needs to be verified.

Set research_needed = "false" if the request:
- Is a generic, self-contained build with no reference to a specific real external product (e.g. "build a
  calculator", "build a to-do app").
- Describes its own requirements fully without needing to know how an external site/product works.

Respond with only "true" or "false".""",
        input_variables=['query']
    )
    chain = prompt | model.with_structured_output(research_need)
    response = chain.invoke({'query': state['new_request']})
    return {'research_needed': response.research_needed}


def checker(state: State) -> State:
    is_research_needed = str(state.get('research_needed', '')).strip().lower()
    if is_research_needed == "true":
        research_result = research.invoke({
            'messages': [HumanMessage(content=state['new_request'])]
        })
        return {'research_context': research_result.get('research_context', '')}
    return {'research_context': ''}


def planner(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are an expert software architect and project planner for an AI code generation system.

Your ONLY job is to analyze the user's request (and, if provided, a research report about
a reference site/product and/or a design schema extracted from a reference image) and produce
a detailed, structured project plan. You do NOT write any code. You only plan.

---

INPUTS YOU WILL RECEIVE:

1. user_query: the user's project request.
2. research_report: findings from a research step about an external site/product the user
   referenced (e.g. "clone Stripe's homepage"). This may be empty — if so, ignore it entirely
   and plan purely from user_query as you normally would.
3. design_schema: a structured extraction of a reference IMAGE the user uploaded — its sections,
   components, exact text, colors, typography, and layout positions. This may say "No reference
   image was provided" — if so, ignore it entirely.

---

HOW TO USE research_report (when present):

- Treat it as REFERENCE MATERIAL describing what the target site/product looks like and how
  it's structured — not as a list of instructions or file names to produce.
- Use it to inform: page sections, layout structure, content/copy style, visual hierarchy,
  key features/components the user likely wants replicated.
- Only use details that are explicitly present in research_report. Do NOT add anything about
  the referenced site from your own general knowledge, even if you recognize it — the report
  is the single source of truth for what was actually observed.
- If research_report is incomplete, partial, or marked as failed/empty for something the user
  asked to clone, do not invent the missing pieces — plan using what's available and keep that
  section simpler/generic rather than guessing specifics.

---

HOW TO USE design_schema (when present):

- design_schema is extracted directly from an image the user wants replicated. It is the SOURCE
  OF TRUTH for exact visual structure — sections, components, exact text content, colors,
  positioning, and typography hierarchy.
- If both design_schema and research_report are present, design_schema wins for anything visual
  or structural (layout, colors, exact text, positioning). research_report should only fill in
  behavioral/functional details the image can't show (e.g. what a button does when clicked).
- Every section_id and component listed in design_schema MUST be reflected as explicit,
  concrete responsibilities in your file plan — not summarized away. Instead of "build the hero
  section," write responsibilities like "hero section: heading text 'X', two buttons labeled
  'Y'/'Z', background color #hex, two-column layout." The code generator only sees your plan,
  not the raw schema, so nothing in the schema is preserved unless you write it down explicitly.
- If the project has multiple files that render visuals (e.g. separate HTML and CSS), the plan
  must specify ONE shared source for colors/typography (e.g. CSS custom properties defined once
  in a single file) so every file stays visually consistent instead of each file guessing
  independently.
- Do not invent sections, components, or colors beyond what design_schema states.

---

PRECEDENCE RULE (when inputs conflict):

user_query (explicit user instructions) > design_schema (visual ground truth from image) >
research_report (general reference material)

Example: if user_query asks for a dark theme but design_schema shows a light-themed reference
image, follow user_query and note the deviation in that section's responsibilities.

---

RULES:

1. Never generate code. Only plan.
2. Be specific about each file's responsibility. Vague descriptions cause bad code.
3. Every file must have a clear, single responsibility. Do not let logic bleed between files.
4. Always specify the exact filename with correct extension.
5. Order files by dependency — files that others import or link to must come first (lower generate_order).
6. If the user does not specify a language, pick the best one based on these rules:
   - Static webpage or portfolio → HTML + CSS + JS (separate files)
   - REST API or backend server → Python (Flask) or Node.js depending on complexity
   - Data processing or AI script → Python
   - CLI tool or automation → Python or Bash
   - System level or performance critical → C++
7. Never combine responsibilities into one file unless the user explicitly asks for a single
   file output.
8. Specify which files link to or import each other explicitly using exact filenames in depends_on.
9. If a project needs a config file (like package.json), include it in the plan.
10. Keep packages minimal — only include what is absolutely necessary.
11. Do not over-engineer simple requests. A calculator script does not need five files.
12. Do not under-build real requests. A portfolio website needs separated HTML, CSS, and JS
    unless told otherwise.
13. If research_report or design_schema describes structural elements (e.g. "hero section,
    pricing grid, footer with 4 columns"), reflect that structure explicitly in the relevant
    file's responsibilities — don't flatten it into a vague "build the homepage" instruction.

---

IMPORTANT BEHAVIOURS:

- The plan you produce is the only instruction the code generator will receive per file. It
  never sees research_report or design_schema directly — if a detail matters, it must appear
  in your responsibilities text, or it will be lost.
- Think about what a senior engineer would consider "complete" for this request — not
  minimal, not over-built.
- Always double check that depends_on relationships are consistent — if app.js depends_on
  index.html's element IDs, make that clear in the responsibilities of both files so the
  code generator keeps them in sync.
- Never let research_report or design_schema override explicit user instructions in user_query.

---

user_query: {query}

research_report: {research_context}

design_schema: {image_description}""",
        input_variables=['query', 'research_context', 'image_description']
    )
    struct_model = model.with_structured_output(all_files)
    chain = prompt | struct_model
    response = chain.invoke({
        'query': state['new_request'],
        'research_context': state['research_context'],
        'image_description': state['image_description'],
    })
    print("Finished planner")
    return {'fileplans': response}


CODE_GEN_TEMPLATE = """You are an expert software engineer generating one file at a time as part of a larger multi-file project.

This is the already generated code of files this one depends on — if empty, do not consider it: {code}
Analyse the blueprint: {blueprint}

Your ONLY job is to generate the complete, correct code for THIS ONE FILE.

---

RULES:

1. Generate code only for the file you are asked to generate. Do not generate or repeat
   code for any other file.
2. Stay strictly within the responsibilities given for this file. Do not add logic that
   belongs to another file.
3. If this file depends on other files, you MUST reference them correctly:
   - Use the exact filenames, function names, IDs, class names, or variable names that
     already exist in the provided dependency code.
   - Do not invent new names that don't match what the dependency files actually expect.
4. If this file is depended upon by files not yet generated, write clean, predictable,
   well-named structures (clear function names, element IDs, exports) so future files can
   correctly reference this one.
5. Include all necessary imports, links, or requires at the top of the file.
6. Add concise inline comments only where logic is non-obvious. Do not over-comment simple
   code.
7. Handle basic edge cases and errors where appropriate for this file's responsibility.
8. Do not include markdown code fences or explanations in the code field — only raw,
   executable code.
9. Follow standard conventions and best practices for the language/file type being
   generated.
10. The code must be complete and functional on its own merit, assuming its dependencies
    exist as described.

---

USING research_context (when provided and non-empty):

11. Use research_context only to inform CONTENT and STRUCTURE fidelity for this specific
    file — actual section names, copy, layout structure, component ordering — when the
    blueprint's responsibilities for this file relate to replicating a referenced site.
12. The blueprint and dependency code remain the source of truth for filenames,
    function names, IDs, and how files link together. research_context never overrides
    those — it only fills in *what the content/structure should look like*, not *how files
    are wired together*.
13. Only use details explicitly present in research_context. If it is empty or doesn't cover
    something this file needs, build that part using standard, sensible conventions instead
    of guessing at the real site's specifics.
14. If research_context is empty, ignore it completely and generate the file purely from
    purpose/responsibilities as normal.

---

USING design_schema (when provided and non-empty):

15. design_schema is an exact extraction from a reference image — treat its bounding boxes,
    colors, text content, and typography as ground truth for THIS file's visual output, not
    as loose inspiration.
16. Reproduce text_content fields EXACTLY character-for-character wherever this file renders
    visible text. Do not paraphrase, shorten, or "improve" the wording.
17. Convert bounding boxes (given as % of image width/height) into responsive layout code —
    percentage widths, flex/grid proportions — never hardcoded pixel positions.
18. Use color hex values from design_schema directly and consistently. If this file is CSS
    and other files also reference these colors, define them once (CSS custom properties)
    rather than repeating raw hex values in multiple places.
19. Map relative typography sizes to concrete values using this fixed scale: largest=2.5rem,
    large=1.75rem, medium=1.25rem, small=1rem, smallest=0.875rem. Do not invent a different scale.
20. For any component where image_description is filled in, render a placeholder that matches
    the described content and approximate aspect ratio — never fabricate a real image URL.
21. Only use details explicitly present in design_schema. If it doesn't cover something this
    file needs, fall back to standard, sensible convention.
22. If design_schema is empty, ignore it completely.

---

WHEN BOTH research_context AND design_schema ARE PRESENT:

23. design_schema governs exact visual details. research_context governs behavior,
    functionality, and structural detail not covered by the image. If they conflict on a
    visual detail, design_schema wins.

---

CRITICAL OUTPUT FORMAT REQUIREMENT:

24. Your response must contain ONLY raw code.
    Do NOT wrap the code in markdown code fences (no ``` at the start or end).
    Do NOT write the language name as the first line.
    Do NOT include any explanation before or after the code.
    The very first character of your response must be actual code.

---

filename: {file}
purpose: {description}
responsibilities: {responsibilities}
depends_on: {depends_on}
package (if required): {package}
research_context (if applicable): {research_context}
design_schema (if applicable): {image_description}"""


def code_generator(state: State) -> State:
    """FIXED:
    - sorts files by generate_order so dependencies are built first
    - only feeds each file the code of its actual depends_on list (not everything)
    - includes 'code' in input_variables
    - passes responsibilities/depends_on through explicitly
    """
    plan = state['fileplans']
    ordered_files = sorted(plan.files, key=lambda f: f.generate_order)

    project_blueprint_ctx = "\n".join([
        f"- File: {f.filename} ({f.language}) [order={f.generate_order}]\n"
        f"  Purpose: {f.purpose}\n"
        f"  Depends on: {f.depends_on}"
        for f in ordered_files
    ])

    code_for_each_file: dict[str, str] = {}

    prompt = PromptTemplate(
        template=CODE_GEN_TEMPLATE,
        input_variables=[
            'blueprint', 'file', 'description', 'responsibilities',
            'depends_on', 'package', 'research_context', 'image_description', 'code'
        ]
    )
    chain = prompt | model | StrOutputParser()

    for f in ordered_files:
        dep_context = "\n\n".join([
            f"--- {dep} ---\n{code_for_each_file[dep]}"
            for dep in f.depends_on
            if dep in code_for_each_file
        ]) or "No dependency code available."

        response = chain.invoke({
            'blueprint': project_blueprint_ctx,
            'file': f.filename,
            'description': f.purpose,
            'responsibilities': "\n".join(f"- {r}" for r in f.responsibilities),
            'depends_on': ", ".join(f.depends_on) if f.depends_on else "none",
            'package': f.package,
            'research_context': state['research_context'],
            'image_description': state['image_description'],
            'code': dep_context,
        })

        code_for_each_file[f.filename] = response

    return {'file_code': code_for_each_file}

# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
graph = StateGraph(State)
graph.add_node('query_optimizer', query_optimizer)
graph.add_node('planner', planner)
graph.add_node('code_generator', code_generator)
graph.add_node('call_researchor_or_not', call_researchor_or_not)
graph.add_node('checker', checker)
graph.add_node('get_image_schema', get_image_schema)
graph.add_node('format_design_schema', format_design_schema)

graph.add_edge(START, 'query_optimizer')
graph.add_edge('query_optimizer', 'call_researchor_or_not')
graph.add_edge('query_optimizer', 'get_image_schema')
graph.add_edge('get_image_schema', 'format_design_schema')
graph.add_edge('format_design_schema', 'planner')
graph.add_edge('call_researchor_or_not', 'checker')
graph.add_edge('checker', 'planner')
graph.add_edge('planner', 'code_generator')
graph.add_edge('code_generator', END)

workflow = graph.compile()

def run_pipeline(prompt: str,reference_image_path:str) -> dict[str, str]:
    """Runs the graph for a given prompt and returns {filename: code}."""
    result = workflow.invoke({'prompt': prompt,'reference_image_path':reference_image_path})
    return result['file_code']

