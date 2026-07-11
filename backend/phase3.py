from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Literal, Callable, TypeVar, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
import base64
import os
import logging

from phase2_planner import research

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ziporg.pipeline")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)
groq_model = ChatGroq(model="llama-3.3-70b-versatile")
coding_model = ChatOpenRouter(model="poolside/laguna-m.1:free")


# ---------------------------------------------------------------------------
# Retry wrapper — every LLM call in this file goes through this instead of a
# bare .invoke(). Exponential backoff, bounded attempts, real error surfaced
# if all retries are exhausted (never silently swallowed).
# ---------------------------------------------------------------------------
T = TypeVar("T")


def invoke_with_retry(
    chain: Any,
    payload: dict,
    node_name: str,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> T:
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:  # noqa: BLE001 - we want to catch and retry broadly here
            last_err = e
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "[%s] attempt %d/%d failed: %s — retrying in %.1fs",
                node_name, attempt, max_attempts, e, delay,
            )
            if attempt < max_attempts:
                time.sleep(delay)
    logger.error("[%s] all %d attempts failed", node_name, max_attempts)
    raise RuntimeError(f"{node_name} failed after {max_attempts} attempts") from last_err


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ResearchSpec(BaseModel):
    project_name: str = Field(description="Short working name for the project")
    project_type: str = Field(description="e.g. 'expense tracker', 'landing page', 'CLI tool'")
    project_summary: str = Field(description="2-3 sentence summary of what this product is")
    target_users: str = Field(description="Who this is built for")
    primary_goal: str = Field(description="The single most important job this product does")
    core_features: List[str] = Field(description="Must-have features, derived from user query + image + research")
    optional_features: List[str] = Field(default_factory=list, description="Nice-to-have features research suggests, only if they naturally fit")
    functional_requirements: List[str] = Field(description="What the product must do")
    non_functional_requirements: List[str] = Field(default_factory=list, description="Performance, accessibility, reliability expectations")
    user_constraints: List[str] = Field(default_factory=list, description="Explicit constraints the user stated")
    research_recommendations: List[str] = Field(default_factory=list, description="Relevant best practices pulled from research")
    design_reference: str = Field(description="Summary of layout, visual style, typography, spacing, color direction, interaction style, responsiveness, major UI components")
    quality_goals: List[str] = Field(default_factory=list, description="What 'done well' looks like for this project")


class DesignToken(BaseModel):
    name: str
    hex: str
    usage: str


class DesignDirection(BaseModel):
    subject_and_audience: str = Field(description="One sentence: what this actually is, for whom")
    signature_element: str = Field(description="The ONE memorable thing this page will be known for")
    palette: List[DesignToken] = Field(description="4-6 named colors")
    display_font: str
    body_font: str
    layout_concept: str = Field(description="One-sentence layout description, not a template name")
    copy_voice: str = Field(description="Tone/register for all written content, e.g. 'plain, active-voice, no filler'")
    avoided_cliches: str = Field(description="Explicitly name which generic AI-design patterns were rejected and why")


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
    relative_size: str = Field(description="largest | large | medium | small | smallest")
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


class fileplan(BaseModel):
    filename: str = Field(description="exact filename with extension")
    purpose: str = Field(description="one clear sentence on what this file does")
    responsibilities: List[str] = Field(description="Specific things this file handles and nothing else")
    depends_on: List[str] = Field(default_factory=list, description="Filenames this file imports, links to, or requires")
    generate_order: int = Field(description="Integer starting from 1. Lower number = generate first.")
    language: str = Field(description="Language this file should be written in")
    package: str = Field(default="", description="Package needed for this file, if any")


class all_files(BaseModel):
    project_type: str = Field(description="e.g. 'static website', 'REST API', 'CLI tool'")
    language: str = Field(description="Primary language/stack chosen for the project")
    reasoning: str = Field(description="One or two sentences explaining why this stack was chosen")
    used_research: bool = Field(description="Whether research_report informed this plan")
    used_design_schema: bool = Field(description="Whether design_schema informed this plan")
    packages: List[str] = Field(default_factory=list, description="Packages needed for the whole project")
    files: List[fileplan]


class State(TypedDict):
    prompt: str
    reference_image_path: str
    new_request: str
    overall_image_design: PageSpec | None
    design_direction: DesignDirection | None
    design_direction_summary: str
    image_description: str
    research_needed: Literal["true", "false"]
    research: ResearchSpec | None
    research_summary: str
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
        input_variables=["query"],
    )
    chain = prompt | groq_model | StrOutputParser()
    response = invoke_with_retry(chain, {"query": state["prompt"]}, "query_optimizer")
    return {"new_request": response}


def get_image_schema(state: State) -> State:
    path = state.get("reference_image_path", "")

    if not path or not os.path.exists(path):
        return {"overall_image_design": None}

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"

    message = HumanMessage(content=[
        {"type": "text", "text": "Give me the complete detail of this image — every section, component, text, color, and layout detail, following the schema."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{image_b64}"},
    ])

    structured_model = model.with_structured_output(PageSpec)
    try:
        result = invoke_with_retry(structured_model, [message], "get_image_schema")
    except RuntimeError:
        logger.error("get_image_schema failed permanently — continuing without a reference image")
        return {"overall_image_design": None}
    return {"overall_image_design": result}


def format_design_schema(state: State) -> State:
    schema = state.get("overall_image_design")
    if not schema:
        return {"image_description": "No reference image was provided."}

    lines = [
        f"Overall layout: {schema.overall_layout}",
        f"Page background: {schema.page_background.hex_estimate} ({schema.page_background.usage})",
        "",
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

    return {"image_description": "\n".join(lines)}


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
  real product, company, or website.
- References specific real-world UI/UX patterns, layouts, or branding that would require knowing what that
  actual site looks like to be built accurately.
- Mentions a specific real API, library, or service whose current behavior/structure needs to be verified.

Set research_needed = "false" if the request:
- Is a generic, self-contained build with no reference to a specific real external product.
- Describes its own requirements fully without needing to know how an external site/product works.

Respond with only "true" or "false".""",
        input_variables=["query"],
    )
    chain = prompt | model.with_structured_output(research_need)
    response = invoke_with_retry(chain, {"query": state["new_request"]}, "call_researchor_or_not")
    return {"research_needed": response.research_needed}


def checker(state: State) -> State:
    is_research_needed = str(state.get("research_needed", "")).strip().lower()
    if is_research_needed == "true":
        try:
            research_result = research.invoke({"messages": [HumanMessage(content=state["new_request"])]})
        except Exception as e:  # noqa: BLE001
            logger.warning("research sub-graph failed: %s — continuing with empty research_context", e)
            return {"research_context": ""}
        return {"research_context": research_result.get("research_context", "")}
    return {"research_context": ""}


def design_direction(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are the design lead at a studio known for giving every client a
distinct visual identity. This client has rejected templated-feeling proposals before.

Ground your choices in the subject itself — its real content, audience, and the page's
single job. Do NOT default to: (1) warm cream background + serif + terracotta accent,
(2) near-black + single neon accent, (3) broadsheet/newspaper hairline-rule layout.
These are the three most common AI-generated design clusters — only use one if the
brief explicitly asks for it.

Pick a signature element: the one thing this page will be remembered by. Keep
everything else disciplined and quiet around it.

Write copy suggestions in active voice, specific to what people actually do here —
never generic marketing filler.

user_query: {query}
research_report: {research_context}
design_schema (if a reference image was provided — treat as ground truth for anything it covers): {image_description}
""",
        input_variables=["query", "research_context", "image_description"],
    )
    struct_model = model.with_structured_output(DesignDirection)
    chain = prompt | struct_model
    response = invoke_with_retry(
        chain,
        {
            "query": state["new_request"],
            "research_context": state["research_context"],
            "image_description": state["image_description"],
        },
        "design_direction",
    )
    return {"design_direction": response}


def format_design_direction(state: State) -> State:
    dd = state.get("design_direction")

    if not dd:
        return {"design_direction_summary": "No design direction was generated."}

    palette = "\n".join(f"  - {t.name}: {t.hex} ({t.usage})" for t in dd.palette)

    summary = f"""Subject/audience: {dd.subject_and_audience}
Signature element: {dd.signature_element}
Palette:
{palette}
Display font: {dd.display_font}
Body font: {dd.body_font}
Layout concept: {dd.layout_concept}
Copy voice: {dd.copy_voice}
Avoided clichés: {dd.avoided_cliches}"""

    return {"design_direction_summary": summary}


def research_gather(state: State) -> State:
    """Merges the optimized query, research_context, and image_description into
    one clean ResearchSpec. This is the single source of truth for WHAT the
    product should be — planner and code_generator both consume its formatted
    summary alongside design_direction_summary."""
    prompt = PromptTemplate(
        template="""You are a Senior Software Requirements Engineer.

Your task is to combine three independent sources of information into one complete RequirementSpec.

You are NOT allowed to generate code, file structures, software architecture, implementation details, or UI designs.

Your responsibility is only to understand the project requirements and produce a single clean specification that every later AI agent can use.

--------------------------------------------------
INPUT 1 : Optimized User Query
--------------------------------------------------
{optimized_query}

--------------------------------------------------
INPUT 2 : Research Report (external site/product findings, may be empty)
--------------------------------------------------
{research_report}

--------------------------------------------------
INPUT 3 : Image Analysis Specification (may say no reference image was provided)
--------------------------------------------------
{image_spec}

--------------------------------------------------
YOUR TASK
--------------------------------------------------
Carefully analyze all three inputs. Merge them into one RequirementSpec.

While merging:
- Preserve every explicit user requirement.
- Use the image specification to understand the intended layout, visual style, UX patterns and component organization.
- Use the research report to improve the project with relevant industry best practices.
- Remove duplicated information.
- Resolve conflicting information intelligently.

Priority order: 1. User Requirements  2. Image Analysis  3. Research Suggestions

Never invent completely unrelated features. If research suggests useful optional features
that naturally fit the project, include them under optional_features.

The RequirementSpec should describe WHAT the application should be. It should NEVER
describe HOW to implement it — do not mention HTML, CSS, JavaScript, frameworks, APIs,
file structure, folder names, database schema, or code. Those are handled by later nodes.

design_reference should summarize: layout, visual style, typography, spacing, color
direction, interaction style, responsiveness, and major UI components.

Return only the structured RequirementSpec.""",
        input_variables=["optimized_query", "research_report", "image_spec"],
    )
    struct_model = model.with_structured_output(ResearchSpec)
    chain = prompt | struct_model
    response = invoke_with_retry(
        chain,
        {
            "optimized_query": state["new_request"],
            "research_report": state["research_context"],
            "image_spec": state["image_description"],
        },
        "research_gather",
    )
    return {"research": response}


def format_research_spec(state: State) -> State:
    spec = state.get("research")
    if not spec:
        return {"research_summary": "No merged requirement spec was generated."}

    def bullets(items: List[str]) -> str:
        return "\n".join(f"  - {i}" for i in items) if items else "  (none)"

    summary = f"""Project: {spec.project_name} ({spec.project_type})
Summary: {spec.project_summary}
Target users: {spec.target_users}
Primary goal: {spec.primary_goal}

Core features:
{bullets(spec.core_features)}

Optional features:
{bullets(spec.optional_features)}

Functional requirements:
{bullets(spec.functional_requirements)}

Non-functional requirements:
{bullets(spec.non_functional_requirements)}

User constraints:
{bullets(spec.user_constraints)}

Research recommendations:
{bullets(spec.research_recommendations)}

Design reference: {spec.design_reference}

Quality goals:
{bullets(spec.quality_goals)}"""

    return {"research_summary": summary}


def planner(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are an expert software architect and project planner for an AI code generation system.

Your ONLY job is to analyze the merged requirement spec (and, if provided, a design schema
extracted from a reference image and a design direction defining this project's visual
identity) and produce a detailed, structured project plan. You do NOT write any code.

---

INPUTS YOU WILL RECEIVE:

1. requirement_spec: the single merged source of truth for WHAT to build — combines the
   user's request, research findings, and image analysis into one clean spec. This is your
   primary input for scope and features.
2. design_schema: a structured extraction of a reference IMAGE the user uploaded — its sections,
   components, exact text, colors, typography, and layout positions. May say "No reference
   image was provided" — if so, ignore it entirely.
3. design_direction_summary: the project's established visual identity — palette, type pairing,
   layout concept, signature element, and copy voice. The source of truth for AESTHETIC
   decisions, the way design_schema is the source of truth for exact structural extraction.
   May say "No design direction was generated" — if so, make reasonable, cohesive choices
   yourself, but keep them consistent across every file in the plan.

---

HOW TO USE requirement_spec:

- Treat core_features and functional_requirements as the mandatory scope of the build.
- Treat optional_features as things to include only if they fit naturally without bloating
  the plan.
- Respect user_constraints as hard limits — never plan around them.
- Use design_reference as a secondary steer on layout/visual style, but design_schema and
  design_direction_summary win over it for anything they explicitly specify (see precedence
  rule below).

---

HOW TO USE design_schema (when present):

- design_schema is extracted directly from an image the user wants replicated. It is the SOURCE
  OF TRUTH for exact visual STRUCTURE — sections, components, exact text content, colors,
  positioning, and typography hierarchy.
- Every section_id and component listed in design_schema MUST be reflected as explicit,
  concrete responsibilities in your file plan — not summarized away. Instead of "build the hero
  section," write responsibilities like "hero section: heading text 'X', two buttons labeled
  'Y'/'Z', background color #hex, two-column layout." The code generator only sees your plan,
  not the raw schema, so nothing in the schema is preserved unless you write it down explicitly.
- Do not invent sections, components, or colors beyond what design_schema states.

---

HOW TO USE design_direction_summary (when present):

- design_direction_summary is the SOURCE OF TRUTH for aesthetic IDENTITY — palette, fonts,
  layout concept, signature element, and copy voice.
- Every responsibility you write for a file that renders visuals or text MUST derive its colors,
  fonts, and copy tone from design_direction_summary — not invent new ones.
- Reflect the signature_element explicitly in the responsibilities of whichever file renders it,
  called out as a distinct, deliberate responsibility.
- Reflect copy_voice as an explicit responsibility wherever a file is expected to write its own
  text content.
- If the project has multiple files that render visuals, the plan must specify ONE shared
  source for colors/typography (e.g. CSS custom properties defined once) so every file stays
  visually consistent instead of each file guessing independently.
- Do not invent a different palette, font pairing, or signature element than what
  design_direction_summary specifies.

---

PRECEDENCE RULE (when inputs conflict):

user_constraints (from requirement_spec) > design_schema (visual ground truth for structure) >
design_direction_summary (visual ground truth for aesthetic identity) > requirement_spec's
design_reference / research_recommendations (general reference material).

For structural conflicts (layout, positioning, exact content), design_schema wins over
design_direction_summary. For aesthetic conflicts (color, type, tone) where design_schema
wasn't extracted from an image detailed enough to specify them, design_direction_summary wins.

---

RULES:

1. Never generate code. Only plan.
2. Be specific about each file's responsibility. Vague descriptions cause bad code.
3. Every file must have a clear, single responsibility. Do not let logic bleed between files.
4. Always specify the exact filename with correct extension.
5. Order files by dependency — files that others import or link to must come first (lower generate_order).
6. If requirement_spec does not make the stack obvious, pick the best one based on these rules:
   - Static webpage or portfolio → HTML + CSS + JS (separate files)
   - REST API or backend server → Python (Flask) or Node.js depending on complexity
   - Data processing or AI script → Python
   - CLI tool or automation → Python or Bash
   - System level or performance critical → C++
7. Never combine responsibilities into one file unless explicitly required.
8. Specify which files link to or import each other explicitly using exact filenames in depends_on.
9. If a project needs a config file (like package.json), include it in the plan.
10. Keep packages minimal — only include what is absolutely necessary.
11. Do not over-engineer simple requests. A calculator script does not need five files.
12. Do not under-build real requests. A portfolio website needs separated HTML, CSS, and JS
    unless told otherwise.
13. If requirement_spec or design_schema describes structural elements, reflect that structure
    explicitly in the relevant file's responsibilities — don't flatten it into a vague
    "build the homepage" instruction.

---

IMPORTANT BEHAVIOURS:

- The plan you produce is the only instruction the code generator will receive per file. It
  never sees requirement_spec, design_schema, or design_direction_summary directly — if a
  detail matters, it must appear in your responsibilities text, or it will be lost.
- Think about what a senior engineer would consider "complete" for this request.
- Always double check that depends_on relationships are consistent.

---

requirement_spec: {research_summary}

design_schema: {image_description}

design_direction_summary: {design_direction_summary}""",
        input_variables=["research_summary", "image_description", "design_direction_summary"],
    )
    struct_model = model.with_structured_output(all_files)
    chain = prompt | struct_model
    response = invoke_with_retry(
        chain,
        {
            "research_summary": state["research_summary"],
            "image_description": state["image_description"],
            "design_direction_summary": state["design_direction_summary"],
        },
        "planner",
    )
    return {"fileplans": response}


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

USING requirement_spec (when provided and non-empty):

11. requirement_spec is the merged source of truth for what the product is and does — use
    it to inform content, copy, and functional behavior for this specific file where the
    blueprint's responsibilities reference project-level features or goals.
12. The blueprint and dependency code remain the source of truth for filenames, function
    names, IDs, and how files link together. requirement_spec never overrides those.

---

USING design_schema (when provided and non-empty):

13. design_schema is an exact extraction from a reference image — treat its bounding boxes,
    colors, text content, and typography as ground truth for THIS file's visual STRUCTURE,
    not as loose inspiration.
14. Reproduce text_content fields EXACTLY character-for-character wherever this file renders
    visible text. Do not paraphrase, shorten, or "improve" the wording.
15. Convert bounding boxes (given as % of image width/height) into responsive layout code —
    percentage widths, flex/grid proportions — never hardcoded pixel positions.
16. Map relative typography sizes to concrete values using this fixed scale: largest=2.5rem,
    large=1.75rem, medium=1.25rem, small=1rem, smallest=0.875rem. Do not invent a different scale.
17. For any component where image_description is filled in, render a placeholder that matches
    the described content and approximate aspect ratio — never fabricate a real image URL.
18. Only use details explicitly present in design_schema. If it doesn't cover something this
    file needs, fall back to design_direction, then to requirement_spec, then to standard
    convention.
19. If design_schema is empty, ignore it completely.

---

USING design_direction (when provided and non-empty):

20. design_direction is the SOURCE OF TRUTH for this project's aesthetic IDENTITY — palette,
    fonts, layout concept, signature element, and copy voice. Every color, font-family, and
    piece of written copy this file produces must trace back to design_direction — never invent
    a new color, font, or generic placeholder copy independently.
21. If this file defines shared visual tokens (e.g. a CSS file with :root custom properties),
    define design_direction's palette and fonts there ONCE, named clearly, so every other file
    references those names instead of repeating raw hex values or font strings.
22. If this file is NOT the shared-tokens file but still renders visuals, reference the shared
    token names/variables defined by its dependency rather than hardcoding raw values again.
23. If this file's responsibilities call out the signature_element, implement it as the single
    most deliberate, polished piece of this file.
24. Wherever this file writes its own visible text, match copy_voice exactly — specific and
    active, never generic marketing filler like "Welcome to our platform."
25. design_schema (when present) governs exact structural extraction; design_direction governs
    aesthetic identity. On conflict, design_schema wins for that specific component.
26. If design_direction is empty, ignore it and fall back to design_schema, then requirement_spec,
    then standard convention.

---

CRITICAL OUTPUT FORMAT REQUIREMENT:

27. Your response must contain ONLY raw code.
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
requirement_spec (if applicable): {research_summary}
design_schema (if applicable): {image_description}
design_direction (if applicable): {design_direction}"""


def code_generator(state: State) -> State:
    """Sorts files by generate_order so dependencies are built first. Only feeds
    each file the code of its actual depends_on list (not everything). Reads
    design_direction_summary and research_summary directly from state — both
    are already-formatted strings computed by earlier nodes."""
    plan = state["fileplans"]
    ordered_files = sorted(plan.files, key=lambda f: f.generate_order)

    project_blueprint_ctx = "\n".join([
        f"- File: {f.filename} ({f.language}) [order={f.generate_order}]\n"
        f"  Purpose: {f.purpose}\n"
        f"  Depends on: {f.depends_on}"
        for f in ordered_files
    ])

    design_direction_ctx = state.get("design_direction_summary", "No design direction was generated.")
    research_summary_ctx = state.get("research_summary", "No merged requirement spec was generated.")

    code_for_each_file: dict[str, str] = {}

    prompt = PromptTemplate(
        template=CODE_GEN_TEMPLATE,
        input_variables=[
            "blueprint", "file", "description", "responsibilities",
            "depends_on", "package", "research_summary", "image_description",
            "design_direction", "code",
        ],
    )
    chain = prompt | coding_model | StrOutputParser()

    for f in ordered_files:
        dep_context = "\n\n".join([
            f"--- {dep} ---\n{code_for_each_file[dep]}"
            for dep in f.depends_on
            if dep in code_for_each_file
        ]) or "No dependency code available."

        response = invoke_with_retry(
            chain,
            {
                "blueprint": project_blueprint_ctx,
                "file": f.filename,
                "description": f.purpose,
                "responsibilities": "\n".join(f"- {r}" for r in f.responsibilities),
                "depends_on": ", ".join(f.depends_on) if f.depends_on else "none",
                "package": f.package,
                "research_summary": research_summary_ctx,
                "image_description": state["image_description"],
                "design_direction": design_direction_ctx,
                "code": dep_context,
            },
            f"code_generator:{f.filename}",
        )

        code_for_each_file[f.filename] = response
        logger.info("Generated %s (%d chars)", f.filename, len(response))

    return {"file_code": code_for_each_file}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
graph = StateGraph(State)
graph.add_node("query_optimizer", query_optimizer)
graph.add_node("call_researchor_or_not", call_researchor_or_not)
graph.add_node("checker", checker)
graph.add_node("get_image_schema", get_image_schema)
graph.add_node("format_design_schema", format_design_schema)
graph.add_node("design_direction", design_direction)
graph.add_node("format_design_direction", format_design_direction)
graph.add_node("research_gather", research_gather)
graph.add_node("format_research_spec", format_research_spec)
graph.add_node("planner", planner)
graph.add_node("code_generator", code_generator)

graph.add_edge(START, "query_optimizer")
graph.add_edge("query_optimizer", "call_researchor_or_not")
graph.add_edge("query_optimizer", "get_image_schema")
graph.add_edge("get_image_schema", "format_design_schema")
graph.add_edge("call_researchor_or_not", "checker")

# design_direction waits on BOTH branches so image_description always exists first
graph.add_edge("format_design_schema", "design_direction")
graph.add_edge("checker", "design_direction")

graph.add_edge("design_direction", "format_design_direction")

# research_gather merges new_request + research_context + image_description into
# one RequirementSpec — runs after format_design_direction so image_description
# and research_context are both finalized.
graph.add_edge("format_design_direction", "research_gather")
graph.add_edge("research_gather", "format_research_spec")
graph.add_edge("format_research_spec", "planner")

graph.add_edge("planner", "code_generator")
graph.add_edge("code_generator", END)

workflow = graph.compile()


def run_pipeline(prompt: str, reference_image_path: str = "") -> dict[str, str]:
    """Runs the graph for a given prompt and returns {filename: code}."""
    result = workflow.invoke({"prompt": prompt, "reference_image_path": reference_image_path})
    return result["file_code"]
