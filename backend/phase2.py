from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Annotated, Literal
from langchain_community.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import operator
import time
from pydantic import BaseModel, Field
import os
import uuid
from phase2_planner import research
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


class research_need(BaseModel):
    research_needed: Literal["true", "false"] = Field(description="whether the research is needed or not")


class fileplan(BaseModel):
    filename: str = Field(description="name of the file that needs to be created for the prompt given by the user")
    description: str = Field(description="description of the file what is it purpose")
    language: str = Field(description="language in which that file should be written")
    package: str = Field(description="package need for that file if needed")


class all_files(BaseModel):
    files: List[fileplan]


class State(TypedDict):
    prompt: str
    new_request: str
    research_needed: Literal["true", "false"]
    research_context: str
    fileplans: all_files
    file_code: dict[str, str]


# defining node
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
    time.sleep(4)

    return {'new_request': response}


# checker to call the research or not
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
    time.sleep(4)

    return {'research_needed': response.research_needed}


def checker(state: State) -> State:
    is_research_needed = str(state.get('research_needed', '')).strip().lower()
    if is_research_needed == "true":
        research_result = research.invoke({
            'messages': [HumanMessage(content=state['new_request'])]
        })
        return {'research_context': research_result.get('research_context', '')}
    else:
        return {'research_context': ''}


# planner node
def planner(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are an expert software architect and project planner for an AI code generation system.

                    Your ONLY job is to analyze the user's request (and, if provided, a research report about
                    a reference site/product) and produce a detailed, structured project plan.
                    You do NOT write any code. You only plan.

                    ---

                    INPUTS YOU WILL RECEIVE:

                    1. user_query: the user's project request.
                    2. research_report: findings from a research step about an external site/product the user
                    referenced (e.g. "clone Stripe's homepage"). This may be empty — if so, ignore it entirely
                    and plan purely from user_query as you normally would.

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
                    - If research_report conflicts with user_query (e.g. user asks for a dark theme but the
                    reference site is light), prioritize the user's explicit instructions in user_query.

                    ---

                    YOUR TASK:

                    Given the inputs above, you must decide:
                    1. What type of project this is
                    2. Which language or tech stack fits best
                    3. Every file needed to build it fully
                    4. The correct order to generate those files
                    5. What each file is responsible for
                    6. All packages/dependencies needed at the project level

                    ---

                    RULES:

                    1. Never generate code. Only plan.
                    2. Be specific about each file's responsibility. Vague descriptions cause bad code.
                    3. Every file must have a clear, single responsibility. Do not let logic bleed between files.
                    4. Always specify the exact filename with correct extension.
                    5. Order files by dependency — files that others import or link to must come first.
                    6. If the user does not specify a language, pick the best one based on these rules:
                    - Static webpage or portfolio → HTML + CSS + JS (separate files)
                    - REST API or backend server → Python (Flask) or Node.js depending on complexity
                    - Data processing or AI script → Python
                    - CLI tool or automation → Python or Bash
                    - System level or performance critical → C++
                    7. Never combine responsibilities into one file unless the user explicitly asks for a single
                    file output.
                    8. Specify which files link to or import each other explicitly using exact filenames.
                    9. If a project needs a config file (like package.json), include it in the plan.
                    10. Keep packages minimal — only include what is absolutely necessary.
                    11. Do not over-engineer simple requests. A calculator script does not need five files.
                    12. Do not under-build real requests. A portfolio website needs separated HTML, CSS, and JS
                        unless told otherwise.
                    13. If research_report describes structural elements (e.g. "hero section, pricing grid,
                        footer with 4 columns"), reflect that structure explicitly in the relevant file's
                        responsibilities — don't flatten it into a vague "build the homepage" instruction.

                    ---

                    OUTPUT FORMAT:

                    You must return a structured output with the following fields:

                    - project_type: what kind of project this is (e.g. "static website", "REST API", "CLI tool")
                    - language: the primary language or stack chosen
                    - reasoning: one or two sentences explaining why you chose this stack
                    - used_research: true/false — whether research_report informed this plan
                    - packages: list of packages needed for the whole project (empty list if none)
                    - files: a list of file objects, each containing:
                        - filename: exact name with extension
                        - purpose: one clear sentence on what this file does
                        - responsibilities: bullet list of specific things this file handles and nothing else
                        - depends_on: list of other filenames this file imports, links to, or requires (empty
                        list if none)
                        - generate_order: integer starting from 1, lower number = generate first

                    ---

                    IMPORTANT BEHAVIOURS:

                    - The plan you produce is the only instruction the code generator will receive per file.
                    If you are vague, the generated code will be wrong. Be precise.
                    - Think about what a senior engineer would consider "complete" for this request — not
                    minimal, not over-built.
                    - Always double check that depends_on relationships are consistent — if app.js depends_on
                    index.html's element IDs, make that clear in the responsibilities of both files so the
                    code generator keeps them in sync.
                    - Never let research_report override explicit user instructions in user_query.

                    ---

                    user_query: {query}

                    research_report: {research_context}""",
        input_variables=['query', 'research_context']
    )
    struct_model = model.with_structured_output(all_files)
    chain = prompt | struct_model
    response = chain.invoke({'query': state['new_request'], 'research_context': state['research_context']})
    time.sleep(4)

    return {'fileplans': response}


# code generator
def code_generator(state: State) -> State:
    all_planned_files = state['fileplans'].files
    filename = [f.filename for f in state['fileplans'].files]
    description = [f.description for f in state['fileplans'].files]
    package = [f.package for f in state['fileplans'].files]

    project_blueprint_ctx = "\n".join([
        f"- File: {f.filename} ({f.language})\n  Purpose: {f.description}"
        for f in all_planned_files
    ])
    code_list = []
    code_for_each_file = {}

    for i in range(0, len(filename)):
        if code_list:
            code_context = "\n\n".join([
                f"--- {filename[j]} ---\n{code_list[j]}"
                for j in range(len(code_list))
            ])
        else:
            code_context = "No files generated yet."

        prompt = PromptTemplate(
            template="""You are an expert software engineer generating one file at a time as part of a larger multi-file project.

                This is the already generated code — if empty, do not consider it: {code}
                Analyse the blueprint: {blueprint}

                You will be given:
                - The overall project context (what the full project is about)
                - The specific file you must generate right now
                - That file's purpose and responsibilities
                - The list of files this file depends on, along with the already-generated code of those
                files (if any)
                - A research report, if the project references an external site/product (may be empty)

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
                12. The blueprint and dependency_context remain the source of truth for filenames,
                    function names, IDs, and how files link together. research_context never overrides
                    those — it only fills in *what the content/structure should look like*, not *how files
                    are wired together*.
                13. Only use details explicitly present in research_context. Do not add anything about the
                    referenced site from general knowledge, even if you recognize it — if research_context
                    is empty or doesn't cover something this file needs, build that part using standard,
                    sensible conventions instead of guessing at the real site's specifics.
                14. If research_context is empty or says no research was conducted, ignore it completely
                    and generate the file purely from purpose/responsibilities/description as normal.

                ---

                CRITICAL OUTPUT FORMAT REQUIREMENT:

                15. Your response must contain ONLY raw code.
                    Do NOT wrap the code in markdown code fences (no ``` at the start or end).
                    Do NOT write the language name as the first line (e.g. do not write "javascript"
                    or "html" or "python" as a standalone word anywhere in your response).
                    Do NOT include any explanation before or after the code.
                    The very first character of your response must be actual code
                    (e.g. "<!DOCTYPE html>", "const", "function", "/* comment */" — not a backtick,
                    not a language name).

                ---

                CONTEXT YOU WILL RECEIVE:

                - project_type: the type of project being built
                - language: the language/stack of the project
                - filename: the exact file you must generate
                - purpose: what this file is supposed to do
                - responsibilities: the specific list of things this file must handle
                - depends_on: filenames this file relies on
                - dependency_context: the actual already-generated code of those dependency files, so you
                can reference correct names
                - research_context: findings about a referenced external site/product, if applicable
                (may be empty)

                ---

                IMPORTANT BEHAVIOURS:

                - If depends_on is empty, generate this file as self-contained.
                - If dependency_context is provided, read it carefully before writing — your file MUST
                correctly link/import/call into it using the exact names already used there.
                - Never assume — only use names, IDs, or functions that actually appear in the
                dependency_context. If something needed isn't there, write the most sensible standard
                convention for it instead of guessing wrong.
                - Never assume real-site specifics not present in research_context either — same rule
                applies: fall back to sensible convention rather than guessing.
                - Your output represents production-quality code. Treat this as if a senior engineer will
                review it before shipping.

                filename is {file} and the description is {description} and the information of the
                package if required {package} and the research context (if applicable) is
                {research_context}""",
            input_variables=['blueprint', 'file', 'description', 'package', 'research_context']
        )

        chain = prompt | model | StrOutputParser()

        response = chain.invoke({
            'blueprint': project_blueprint_ctx,
            'file': filename[i],
            'description': description[i],
            'package': package[i],
            'research_context': state['research_context'],
            'code': code_context
        })

        code_list.append(response)
        code_for_each_file[filename[i]] = response
    return {'file_code': code_for_each_file}

graph = StateGraph(State)
graph.add_node('query_optimizer', query_optimizer)
graph.add_node('planner', planner)
graph.add_node('code_generator', code_generator)
graph.add_node('call_researchor_or_not', call_researchor_or_not)
graph.add_node('checker', checker)

graph.add_edge(START, 'query_optimizer')
graph.add_edge('query_optimizer', 'call_researchor_or_not')
graph.add_edge('call_researchor_or_not', 'checker')
graph.add_edge('checker', 'planner')
graph.add_edge('planner', 'code_generator')
graph.add_edge('code_generator', END)

workflow = graph.compile()

def run_pipeline(prompt: str) -> dict[str, str]:
    """Runs the graph for a given prompt and returns {filename: code}."""
    result = workflow.invoke({'prompt': prompt})
    return result['file_code']

# if __name__ == "__main__":
#     response = workflow.invoke({
#         'prompt': "Build me the working calculator app like the one on the Apple website, with a similar layout and design, but with additional features for scientific calculations."
#     })
#     print(response['research_needed'])
