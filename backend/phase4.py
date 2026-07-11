from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
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
# Schemas
# ---------------------------------------------------------------------------

class ColorToken(BaseModel):
    name: str
    value: str
    usage: str


class TypographyToken(BaseModel):
    role: str
    font: str
    weight: str
    size: str


class DesignSystemSpec(BaseModel):
    colors: List[ColorToken]
    typography: List[TypographyToken]
    spacing_scale: List[int]
    border_radius: str
    shadows: List[str]
    icon_style: str
    animation_style: str
    grid_system: str


class RequirementSpec(BaseModel):
    project_name: str
    project_type: str
    project_summary: str

    target_users: str
    primary_goal: str

    core_features: List[str]
    optional_features: List[str]

    functional_requirements: List[str]
    non_functional_requirements: List[str]

    constraints: List[str]

    research_summary: str | None
    design_summary: str | None


class ProductSpec(BaseModel):
    product_category: str
    personality: str
    tone: str
    emotional_goal: str
    design_keywords: List[str]
    differentiation: str
    branding_summary: str


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


# --- NEW: UX Spec ------------------------------------------------------

class UserFlow(BaseModel):
    flow_name: str = Field(description="e.g. 'First-time signup', 'Checkout'")
    entry_point: str = Field(description="Where the user starts this flow")
    steps: List[str] = Field(description="Ordered, concrete steps the user takes")
    exit_point: str = Field(description="What success/completion looks like")
    failure_paths: List[str] = Field(description="What happens if something goes wrong mid-flow")


class ScreenSpec(BaseModel):
    screen_id: str = Field(description="Short unique id, e.g. 'home', 'dashboard', 'settings'")
    screen_name: str
    purpose: str = Field(description="The single job this screen does")
    key_elements: List[str] = Field(description="The must-have elements/sections on this screen")
    navigates_to: List[str] = Field(description="screen_ids this screen can navigate to")
    entry_from: List[str] = Field(description="screen_ids or events that lead here")


class UXSpec(BaseModel):
    information_architecture: str = Field(description="How screens/pages relate to each other, in prose")
    navigation_pattern: str = Field(description="e.g. 'top nav + breadcrumbs', 'bottom tab bar', 'sidebar nav'")
    screens: List[ScreenSpec]
    user_flows: List[UserFlow]
    responsive_behavior: str = Field(description="How layout/navigation adapts across breakpoints")
    accessibility_requirements: List[str] = Field(description="Concrete a11y requirements: contrast, keyboard nav, ARIA, focus order, etc.")
    empty_states: List[str] = Field(description="What each key empty/zero-data state should say and offer")
    error_handling_ux: str = Field(description="How errors are surfaced to the user across the product")


# --- NEW: Component Spec ------------------------------------------------

class ComponentVariant(BaseModel):
    variant_name: str = Field(description="e.g. 'primary', 'secondary', 'destructive'")
    description: str
    when_to_use: str


class ComponentStateSpec(BaseModel):
    state_name: str = Field(description="e.g. 'default', 'hover', 'disabled', 'loading', 'error'")
    visual_change: str = Field(description="Concrete visual difference from default state, using design_system tokens")


class ComponentDef(BaseModel):
    component_name: str = Field(description="e.g. 'Button', 'InputField', 'Card', 'NavBar'")
    component_type: str = Field(description="atom | molecule | organism (atomic design tiering)")
    purpose: str
    props: List[str] = Field(description="Configurable inputs, e.g. 'label', 'variant', 'disabled', 'icon'")
    variants: List[ComponentVariant]
    states: List[ComponentStateSpec]
    composition_notes: str = Field(description="What smaller components this is built from, if any; spacing/sizing rules referencing design_system tokens")


class ComponentSpec(BaseModel):
    components: List[ComponentDef]
    naming_convention: str = Field(description="e.g. 'PascalCase component files, kebab-case CSS classes'")
    reuse_strategy: str = Field(description="How components should be composed/shared across screens to avoid duplication")


# --- NEW: Interaction Spec ----------------------------------------------

class InteractionRule(BaseModel):
    trigger: str = Field(description="e.g. 'click', 'hover', 'form submit', 'scroll into view'")
    element: str = Field(description="Which component/element this applies to")
    behavior: str = Field(description="What happens, concretely")
    feedback: str = Field(description="Visual/haptic/textual feedback given to the user")
    timing: str = Field(description="Duration/easing, e.g. '150ms ease-out'")


class InteractionSpec(BaseModel):
    micro_interactions: List[InteractionRule]
    form_validation_behavior: str = Field(description="When validation runs (on blur/submit/live), how errors are shown")
    loading_states: str = Field(description="Skeletons vs spinners vs progress bars, and when each is used")
    error_states: str = Field(description="How component-level and page-level errors are visually communicated")
    transition_style: str = Field(description="Page/route transition behavior")
    gesture_support: str = Field(description="Touch gestures supported, if any (swipe, drag, pinch); 'none' if not applicable")


# --- Planning schemas -----------------------------------------------------

class fileplan(BaseModel):
    filename: str = Field(description="exact filename with extension")
    purpose: str = Field(description="one clear sentence on what this file does")
    responsibilities: List[str] = Field(
        description="Specific things this file handles and nothing else — including exact "
                    "text, colors, and layout details from design_schema/design_direction where relevant"
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


class State(TypedDict):
    prompt: str
    reference_image_path: str
    new_request: str
    overall_image_design: PageSpec | None
    design_direction: DesignDirection | None
    design_direction_summary: str
    image_description: str
    research_needed: Literal["true", "false"]
    research: RequirementSpec
    product_specification: ProductSpec
    design_system: DesignSystemSpec | None
    design_system_summary: str
    ux_spec: UXSpec | None
    ux_summary: str
    component_spec: ComponentSpec | None
    component_summary: str
    interaction_spec: InteractionSpec | None
    interaction_summary: str
    research_context: str
    fileplans: all_files
    file_code: dict[str, str]


groq_model = ChatGroq(model="llama-3.3-70b-versatile")
coding_model = ChatOpenRouter(model="poolside/laguna-m.1:free")

DESIGN_SYSTEM_TEMPLATE = """You are the design systems lead at a studio that builds durable, reusable
visual languages — not one-off page styles. Your job is to convert a requirement spec and a product's
strategic identity into a complete set of DESIGN TOKENS: the atomic visual rules every component,
page, and future designer on this project will draw from.

You are NOT designing a page. You are NOT writing HTML or CSS. You are defining the vocabulary that
HTML/CSS will later be written in. If you find yourself describing a layout, a section, or a specific
component, stop — that belongs to a later step, not here.

---
INPUT: Requirement Specification
project_name: {project_name}
project_type: {project_type}
project_summary: {project_summary}
target_users: {target_users}
core_features: {core_features}
non_functional_requirements: {non_functional_requirements}
constraints: {constraints}

INPUT: Product Specification (strategic identity)
product_category: {product_category}
personality: {personality}
tone: {tone}
emotional_goal: {emotional_goal}
design_keywords: {design_keywords}
differentiation: {differentiation}
branding_summary: {branding_summary}

---
INSTRUCTIONS

1. COLORS
   - Derive a color system that expresses personality, tone, and emotional_goal — not a generic
     "safe" palette. design_keywords must be visibly reflected in your hue, saturation, and
     contrast choices.
   - Include at minimum: a primary brand color, a secondary/accent color, a neutral/background
     scale (at least 2 neutrals — light and dark, or a background + surface), a text color
     (and secondary text color if the product needs hierarchy), and a semantic state color
     (success, error, or warning — whichever is relevant to core_features).
   - Every color must have a clear, specific `usage` — not "general use." E.g. "primary CTA
     buttons and active nav state," not "accent color."
   - Avoid the three most common AI-default palettes unless target_users/product_category
     explicitly calls for one: (a) cream background + terracotta/rust accent, (b) near-black +
     single neon accent, (c) generic blue-and-white SaaS palette. Justify your palette choice
     through personality/tone, don't default to convention.
   - Give every token a real, memorable `name` (e.g. "Ink", "Signal Coral", "Paper") not "color1."

2. TYPOGRAPHY
   - Define a role for every text purpose the product will actually need, inferred from
     core_features and project_type — at minimum: a display/heading role and a body role.
     Add more roles (caption, label, button, code/mono if relevant) only if core_features or
     project_type justify them — do not pad the list.
   - Font choices must reflect tone and copy_voice implied by personality — a playful consumer
     app and an enterprise B2B tool should never converge on the same type pairing by default.
   - Use real, specific font names (Google Fonts or system-safe stacks), never "a modern
     sans-serif."
   - size should be a concrete value (rem or px), not relative ("large"). weight should be a
     real numeric or named weight (e.g. "600" or "semibold"), not "bold-ish."

3. SPACING SCALE
   - Provide a single consistent numeric scale (in px) that every future component will use for
     margin/padding/gap — typically 6-9 values, often following a ratio (4px base, 8px base, or
     a modular scale). Do not provide arbitrary unrelated numbers.
   - The scale's density (tight vs. spacious) should reflect spacing implications from
     product_category and target_users (e.g. a data-dense dashboard needs a tighter base unit
     than a premium marketing site).

4. BORDER RADIUS
   - Choose ONE consistent radius value or a small system (e.g. "sm/md/lg" as one string) that
     matches personality — sharp/zero radius reads differently than heavily rounded. Do not
     default to a generic "8px" without justification from tone/personality.

5. SHADOWS
   - Define 2-4 shadow levels (e.g. subtle, medium, elevated) as real CSS-compatible shadow
     values, calibrated to match the product's emotional_goal — a "trustworthy, premium" product
     usually wants soft, low-contrast shadows; a "bold, energetic" product can support harder,
     more graphic shadows. Do not just output generic Bootstrap-style shadow values without
     reasoning about fit.

6. ICON STYLE
   - Specify a concrete icon style (e.g. "outlined, 1.5px stroke, rounded caps" or "solid/filled,
     geometric") consistent with the typography weight and border_radius decisions — icon style,
     type weight, and corner treatment should feel like they belong to the same product.

7. ANIMATION STYLE
   - Describe the motion personality in concrete terms a developer can implement — easing
     (e.g. "ease-out, 200ms"), what kind of interactions get motion (hover, page transition,
     loading states), and restraint level. Ground this in emotional_goal and non_functional_requirements
     (e.g. accessibility/performance constraints may mean motion should be minimal or
     respect prefers-reduced-motion).

8. GRID SYSTEM
   - Specify a concrete, implementable grid (e.g. "12-column, 24px gutter, 1200px max-width
     container" or "CSS Grid, 4/8/12 column responsive breakpoints at 640/1024/1440px").
     Match density/complexity to project_type (a dashboard needs a more granular grid than a
     single-page landing site).

---
CONSISTENCY RULES
- Every decision must trace back to personality, tone, emotional_goal, or design_keywords from
  the Product Specification — this is not a free-form aesthetic exercise, it's a translation of
  strategic identity into implementable tokens.
- All tokens must feel like they belong to the same product. Do not let color palette, type
  pairing, radius, shadows, icon style, and motion feel like they were chosen independently of
  each other.
- Do not include layout, page structure, component composition, or any HTML/CSS. Tokens only.
- Do not invent tokens beyond what the schema asks for.

---
OUTPUT FORMAT
Return ONLY a valid JSON object (no markdown fences, no commentary) matching exactly this schema:
{{
  "colors": [{{"name": string, "value": string, "usage": string}}, ...],
  "typography": [{{"role": string, "font": string, "weight": string, "size": string}}, ...],
  "spacing_scale": [int, ...],
  "border_radius": string,
  "shadows": [string, ...],
  "icon_style": string,
  "animation_style": string,
  "grid_system": string
}}
"""

UX_SPEC_TEMPLATE = """You are a senior UX architect. Your job is to convert a requirement spec and
product identity into a concrete UX SPECIFICATION: the screens, navigation, flows, and edge-case
behavior every later design/dev step will build against.

You are NOT choosing colors, fonts, or component visuals — that belongs to the design system and
component spec. You ARE deciding what screens exist, how a user moves between them, and what
happens when things go wrong or are empty.

---
INPUT: Requirement Specification
project_name: {project_name}
project_type: {project_type}
project_summary: {project_summary}
target_users: {target_users}
primary_goal: {primary_goal}
core_features: {core_features}
optional_features: {optional_features}
functional_requirements: {functional_requirements}
non_functional_requirements: {non_functional_requirements}

INPUT: Product Specification (strategic identity)
product_category: {product_category}
personality: {personality}
emotional_goal: {emotional_goal}

---
INSTRUCTIONS

1. SCREENS — Enumerate every screen/page this project actually needs to fulfil core_features and
   primary_goal. Do not invent screens core_features don't justify; do not omit a screen a listed
   feature clearly requires. Each screen needs a single, clear purpose.
2. NAVIGATION — Decide one concrete navigation_pattern appropriate to project_type and target_users
   (e.g. a data tool suits a persistent sidebar; a marketing site suits a simple top nav). Wire every
   screen's navigates_to/entry_from so the whole screen graph is connected and reachable.
3. USER FLOWS — Write out the 2-5 most important user_flows implied by core_features and
   primary_goal, as ordered, concrete steps (not abstractions like "user completes onboarding" —
   say what they actually see and do at each step). Include realistic failure_paths, not just the
   happy path.
4. RESPONSIVE BEHAVIOR — State concretely how navigation and layout change across breakpoints,
   grounded in non_functional_requirements if they mention device targets.
5. ACCESSIBILITY — List concrete, testable accessibility_requirements (keyboard navigability, focus
   order, color contrast minimums, ARIA roles/labels for interactive elements) relevant to
   core_features — not a generic boilerplate list.
6. EMPTY STATES — For each screen that can show zero data (e.g. no results, no items yet, first
   login), specify what it says and what action it offers, in copy that matches personality/tone.
7. ERROR HANDLING — Describe one consistent, product-wide pattern for surfacing errors (inline vs.
   toast vs. modal, etc.) rather than a different pattern per screen.

---
CONSISTENCY RULES
- Every screen and flow must trace back to core_features, functional_requirements, or primary_goal.
  Do not add screens/flows unrelated to what was actually requested.
- The screen graph (navigates_to / entry_from) must be internally consistent — every screen_id
  referenced must also exist as its own ScreenSpec entry.
- Keep scope proportional to project_type — a simple tool needs a simple screen graph; do not
  over-architect.

---
user_query (for grounding, informational only): {query}

OUTPUT FORMAT
Return ONLY a valid JSON object (no markdown fences, no commentary) matching exactly this schema:
{{
  "information_architecture": string,
  "navigation_pattern": string,
  "screens": [{{"screen_id": string, "screen_name": string, "purpose": string, "key_elements": [string, ...], "navigates_to": [string, ...], "entry_from": [string, ...]}}, ...],
  "user_flows": [{{"flow_name": string, "entry_point": string, "steps": [string, ...], "exit_point": string, "failure_paths": [string, ...]}}, ...],
  "responsive_behavior": string,
  "accessibility_requirements": [string, ...],
  "empty_states": [string, ...],
  "error_handling_ux": string
}}
"""

COMPONENT_SPEC_TEMPLATE = """You are a senior design-systems/frontend architect. Your job is to define
the REUSABLE COMPONENT LIBRARY for this product — the discrete building blocks every screen will be
assembled from — grounded in the UX spec's screens and the design system's tokens.

You are NOT laying out pages (that's the file/plan step) and NOT re-defining colors/fonts (that's
already fixed in design_system — reference it, don't reinvent it).

---
INPUT: UX Spec (screens this library must support)
screens: {screens}
key_elements_by_screen: {key_elements}

INPUT: Design System (tokens this library must use)
colors: {colors}
typography: {typography}
spacing_scale: {spacing_scale}
border_radius: {border_radius}
icon_style: {icon_style}

INPUT: Product identity
personality: {personality}
design_keywords: {design_keywords}

---
INSTRUCTIONS

1. Derive the component list DIRECTLY from key_elements_by_screen — every recurring UI element
   across screens (buttons, inputs, cards, nav items, etc.) becomes exactly one ComponentDef, reused
   everywhere it appears rather than redefined per screen.
2. For each component, list only the props it genuinely needs to support its variants/states across
   all the screens that use it.
3. variants should reflect real distinctions this product needs (e.g. "primary"/"secondary" only if
   the UX spec's flows actually call for both) — do not pad with unused variants.
4. states must cover at minimum default and disabled for interactive components, plus loading/error
   where the UX spec's error_handling_ux or forms imply it.
5. composition_notes must reference design_system's actual token names/values (spacing_scale steps,
   border_radius, colors) — every size/spacing/color decision here must be traceable to a token
   already defined, never a new arbitrary value.
6. Keep component_type as atomic-design tiering (atom/molecule/organism) so dependency order between
   components is clear (atoms have no sub-components; organisms compose molecules/atoms).
7. naming_convention and reuse_strategy must be concrete enough for a code generator to follow
   mechanically, not general engineering advice.

---
CONSISTENCY RULES
- Every component must be traceable to at least one screen's key_elements.
- Do not invent visual decisions the design_system doesn't already provide — reference its tokens.
- Do not include page-specific layout/composition of multiple screens together — one component's
  definition, not how screens arrange components.

---
OUTPUT FORMAT
Return ONLY a valid JSON object (no markdown fences, no commentary) matching exactly this schema:
{{
  "components": [{{"component_name": string, "component_type": string, "purpose": string, "props": [string, ...], "variants": [{{"variant_name": string, "description": string, "when_to_use": string}}, ...], "states": [{{"state_name": string, "visual_change": string}}, ...], "composition_notes": string}}, ...],
  "naming_convention": string,
  "reuse_strategy": string
}}
"""

INTERACTION_SPEC_TEMPLATE = """You are a senior interaction designer. Your job is to define exactly how
this product BEHAVES moment-to-moment — the micro-interactions, feedback, and state transitions that
make the component library and UX flows feel alive, grounded in what's already been defined.

You are NOT redefining components or screens — you are specifying their behavior under user action.

---
INPUT: Component Spec (what exists and what states it supports)
components: {components}

INPUT: UX Spec (flows and error handling this must support)
user_flows: {user_flows}
error_handling_ux: {error_handling_ux}
empty_states: {empty_states}

INPUT: Design System (motion/animation baseline already established)
animation_style: {animation_style}

INPUT: Product identity
personality: {personality}
emotional_goal: {emotional_goal}

---
INSTRUCTIONS

1. For each interactive component (from components) that plausibly needs one, write a concrete
   InteractionRule: exact trigger, the element it applies to (by component_name), the resulting
   behavior, the feedback given to the user, and timing consistent with animation_style. Do not
   write a rule for purely static/presentational components.
2. form_validation_behavior must state precisely when validation runs and how/where errors appear,
   consistent with error_handling_ux already defined in the UX spec — do not contradict it.
3. loading_states must specify which loading pattern (skeleton/spinner/progress bar) applies to which
   kind of operation (page load vs. action vs. background fetch), matching component states already
   defined for "loading" where applicable.
4. error_states must extend error_handling_ux with the specific visual treatment at the component
   level (e.g. red border + inline message under InputField on validation failure).
5. transition_style must be consistent with animation_style's easing/duration baseline — do not
   introduce a different motion language.
6. gesture_support: only include real gestures if target usage plausibly involves touch; otherwise
   state "none" explicitly rather than inventing gestures for a desktop-only tool.

---
CONSISTENCY RULES
- Every InteractionRule's `element` must be a component_name that already exists in the component
  spec — do not reference undefined components.
- Timing/easing across all interactions must feel like one motion language, matching animation_style.
- Do not duplicate what error_handling_ux/empty_states already say — extend them with concrete,
  component-level detail instead.

---
OUTPUT FORMAT
Return ONLY a valid JSON object (no markdown fences, no commentary) matching exactly this schema:
{{
  "micro_interactions": [{{"trigger": string, "element": string, "behavior": string, "feedback": string, "timing": string}}, ...],
  "form_validation_behavior": string,
  "loading_states": string,
  "error_states": string,
  "transition_style": string,
  "gesture_support": string
}}
"""


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
    chain = prompt | groq_model | StrOutputParser()
    response = chain.invoke({'query': state['prompt']})
    time.sleep(4)
    return {'new_request': response}


def get_image_schema(state: State) -> State:
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

    structured_model = groq_model.with_structured_output(PageSpec)
    result = structured_model.invoke([message])
    return {'overall_image_design': result}


def format_design_schema(state: State) -> State:
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
    chain = prompt | groq_model.with_structured_output(research_need)
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
    return {'research_context': ''}


def requirements(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are a requirements analyst AI. Your job is to read a user's product/project query 
(and optionally some supporting research notes and/or an image summary) and produce a 
complete, well-structured requirement specification as a single JSON object.

## Inputs

**User Query (required):**
{query}

**Research Summary (optional — may be empty):**
{research_summary}

**Image Summary (optional — may be empty):**
{image_summary}

## Instructions

1. Read the user query carefully to understand what they want to build.
2. If a research summary is provided, use it to inform feasibility, feature scope, 
   competitive context, and technical constraints. Fold its key implications into 
   the relevant fields (do not just repeat it verbatim).
3. If an image summary is provided (e.g. a description of a sketch, mockup, or 
   reference screenshot), treat it as a strong signal for UI/UX expectations, 
   layout, and feature hints. Reflect this in core_features and functional_requirements.
4. If research_summary or image_summary are missing/empty, proceed using only the 
   query — do not invent fake research or image content, and set those output 
   fields to null.
5. Infer reasonable details when the query is underspecified, but stay grounded 
   in what was actually said — do not hallucinate unrelated features.
6. Be specific and actionable. Avoid vague filler like "should be user-friendly" 
   without elaborating what that means concretely.

## Output Format

Return ONLY a valid JSON object (no markdown fences, no commentary) matching 
exactly this schema:

{{
  "project_name": string,
  "project_type": string,
  "project_summary": string,
  "target_users": string,
  "primary_goal": string,
  "core_features": [string, ...],
  "optional_features": [string, ...],
  "functional_requirements": [string, ...],
  "non_functional_requirements": [string, ...],
  "constraints": [string, ...],
  "research_summary": string or null,
  "design_summary": string or null
}}

## Rules
- Every string field must be non-empty except research_summary and design_summary, 
  which are null when no corresponding input was provided.
- Every list must contain at least 2 items when the query gives enough information; 
  otherwise return your best reasonable inference — never an empty list unless truly 
  nothing applies.
- Do not include any text outside the JSON object.""",
        input_variables=['query', 'image_summary', 'research_summary']
    )
    struct_model = model.with_structured_output(RequirementSpec)
    chain = prompt | struct_model
    response = chain.invoke({
        'query': state['new_request'],
        'image_summary': state['image_description'],
        'research_summary': state['research_context']
    })
    return {'research': response}


def product_spec(state: State) -> State:
    req = state.get('research')
    prompt = PromptTemplate(
        template="""You are a Product Strategist AI. Your job is to take a technical requirement 
specification and give the product an IDENTITY — deciding what it should feel 
like, who it's really for, and how it should be perceived, above and beyond its 
literal feature list.

Your job is NOT to describe what the product does (that's already been captured). 
Your job is to decide what kind of product this is in the market, and what 
emotional/aesthetic identity will make it land.

Examples of the transformation you must perform:
- "Expense Tracker" → "Premium Finance Product"
- "Portfolio" → "Creative AI Engineer Brand"
- "CRM" → "Enterprise SaaS"

## Input: Requirement Specification

Project Name: {project_name}
Project Type: {project_type}
Project Summary: {project_summary}
Target Users: {target_users}
Primary Goal: {primary_goal}
Core Features: {core_features}
Optional Features: {optional_features}
Functional Requirements: {functional_requirements}
Non-Functional Requirements: {non_functional_requirements}
Constraints: {constraints}
Research Summary: {research_summary}
Design Summary: {design_summary}

## Instructions

1. Read between the lines of the requirement spec. Two projects with identical 
   features can have completely different identities depending on who they're 
   for and what feeling they need to evoke — decide which one this is.
2. Elevate the framing. Do not just restate project_type — reposition it into 
   a market category that signals ambition and positioning (e.g. a scheduling 
   tool for therapists might become a "Boutique Wellness Practice Platform," 
   not just "Booking App").
3. If research_summary or design_summary are provided (they may be null/empty), 
   use them to sharpen tone and differentiation — e.g. competitor positioning 
   from research, or visual cues from design notes. If they are null, rely 
   solely on the requirement spec and make well-reasoned inferences.
4. Ensure personality, tone, and emotional_goal are distinct from one another:
   - personality = the character/persona the product has, as if it were a person
   - tone = how it communicates (voice, language, visual register)
   - emotional_goal = the specific feeling the user should walk away with
5. design_keywords should be concrete, usable directly by a designer — adjectives 
   and stylistic descriptors, not vague abstractions.
6. differentiation should state what makes this product distinct from generic/
   competing alternatives in its category — grounded in the actual requirements, 
   not invented hype.
7. branding_summary should synthesize everything above into a short, cohesive 
   paragraph a designer or brand strategist could act on immediately.

## Output Format

Return ONLY a valid JSON object (no markdown fences, no commentary) matching 
exactly this schema:

{{
  "product_category": string,
  "personality": string,
  "tone": string,
  "emotional_goal": string,
  "design_keywords": [string, ...],
  "differentiation": string,
  "branding_summary": string
}}

## Rules
- Every field must be non-empty.
- design_keywords must contain 3-6 concrete, specific words (not phrases), e.g. 
  ["Professional", "Premium", "Minimal", "Confident", "Reliable"].
- Do not include any text outside the JSON object.
- Do not simply restate the project_type or project_summary — every field must 
  add strategic/emotional value beyond what the requirement spec already states.""",
        input_variables=['project_name', 'project_type', 'project_summary', 'target_users', 'primary_goal',
                          'core_features', 'optional_features', 'functional_requirements',
                          'non_functional_requirements', 'constraints', 'research_summary', 'design_summary']
    )
    struct_model = model.with_structured_output(ProductSpec)
    chain = prompt | struct_model
    response = chain.invoke({
        'project_name': req.project_name,
        'project_type': req.project_type,
        'project_summary': req.project_summary,
        'target_users': req.target_users,
        'primary_goal': req.primary_goal,
        'core_features': req.core_features,
        'optional_features': req.optional_features,
        'functional_requirements': req.functional_requirements,
        'non_functional_requirements': req.non_functional_requirements,
        'constraints': req.constraints,
        'research_summary': req.research_summary,
        'design_summary': req.design_summary,
    })
    return {'product_specification': response}


def design_system(state: State) -> State:
    req = state.get('research')
    product = state.get('product_specification')
    prompt = PromptTemplate(template=DESIGN_SYSTEM_TEMPLATE, input_variables=[
        'project_name', 'project_type', 'project_summary', 'target_users', 'core_features',
        'non_functional_requirements', 'constraints', 'product_category', 'personality',
        'tone', 'emotional_goal', 'design_keywords', 'differentiation', 'branding_summary'
    ])
    struct_model = model.with_structured_output(DesignSystemSpec)
    chain = prompt | struct_model
    response = chain.invoke({
        'project_name': req.project_name,
        'project_type': req.project_type,
        'project_summary': req.project_summary,
        'target_users': req.target_users,
        'core_features': req.core_features,
        'non_functional_requirements': req.non_functional_requirements,
        'constraints': req.constraints,
        'product_category': product.product_category,
        'personality': product.personality,
        'tone': product.tone,
        'emotional_goal': product.emotional_goal,
        'design_keywords': product.design_keywords,
        'differentiation': product.differentiation,
        'branding_summary': product.branding_summary,
    })
    time.sleep(4)
    return {'design_system': response}


def format_design_system(state: State) -> State:
    ds = state.get('design_system')
    if not ds:
        return {'design_system_summary': "No design system was generated."}

    colors = "\n".join(f"  - {c.name}: {c.value} ({c.usage})" for c in ds.colors)
    typography = "\n".join(
        f"  - {t.role}: {t.font}, weight={t.weight}, size={t.size}" for t in ds.typography
    )
    shadows = "\n".join(f"  - {s}" for s in ds.shadows)

    summary = f"""Colors:
{colors}
Typography:
{typography}
Spacing scale: {ds.spacing_scale}
Border radius: {ds.border_radius}
Shadows:
{shadows}
Icon style: {ds.icon_style}
Animation style: {ds.animation_style}
Grid system: {ds.grid_system}"""
    return {'design_system_summary': summary}


def ux_spec_node(state: State) -> State:
    req = state.get('research')
    product = state.get('product_specification')
    prompt = PromptTemplate(template=UX_SPEC_TEMPLATE, input_variables=[
        'project_name', 'project_type', 'project_summary', 'target_users', 'primary_goal',
        'core_features', 'optional_features', 'functional_requirements', 'non_functional_requirements',
        'product_category', 'personality', 'emotional_goal', 'query'
    ])
    struct_model = model.with_structured_output(UXSpec)
    chain = prompt | struct_model
    response = chain.invoke({
        'project_name': req.project_name,
        'project_type': req.project_type,
        'project_summary': req.project_summary,
        'target_users': req.target_users,
        'primary_goal': req.primary_goal,
        'core_features': req.core_features,
        'optional_features': req.optional_features,
        'functional_requirements': req.functional_requirements,
        'non_functional_requirements': req.non_functional_requirements,
        'product_category': product.product_category,
        'personality': product.personality,
        'emotional_goal': product.emotional_goal,
        'query': state['new_request'],
    })
    time.sleep(4)
    return {'ux_spec': response}


def format_ux_spec(state: State) -> State:
    ux = state.get('ux_spec')
    if not ux:
        return {'ux_summary': "No UX spec was generated."}

    screens = "\n".join(
        f"  - [{s.screen_id}] {s.screen_name}: {s.purpose}\n"
        f"    key_elements: {s.key_elements}\n"
        f"    navigates_to: {s.navigates_to} | entry_from: {s.entry_from}"
        for s in ux.screens
    )
    flows = "\n".join(
        f"  - {f.flow_name}: entry={f.entry_point} -> steps={f.steps} -> exit={f.exit_point}\n"
        f"    failure_paths: {f.failure_paths}"
        for f in ux.user_flows
    )
    summary = f"""Information architecture: {ux.information_architecture}
Navigation pattern: {ux.navigation_pattern}
Screens:
{screens}
User flows:
{flows}
Responsive behavior: {ux.responsive_behavior}
Accessibility requirements: {ux.accessibility_requirements}
Empty states: {ux.empty_states}
Error handling UX: {ux.error_handling_ux}"""
    return {'ux_summary': summary}


def component_spec_node(state: State) -> State:
    ux = state.get('ux_spec')
    ds = state.get('design_system')
    product = state.get('product_specification')

    screens = [s.screen_name for s in ux.screens] if ux else []
    key_elements = {s.screen_name: s.key_elements for s in ux.screens} if ux else {}
    colors = [f"{c.name}:{c.value}" for c in ds.colors] if ds else []
    typography = [f"{t.role}:{t.font}/{t.weight}/{t.size}" for t in ds.typography] if ds else []

    prompt = PromptTemplate(template=COMPONENT_SPEC_TEMPLATE, input_variables=[
        'screens', 'key_elements', 'colors', 'typography', 'spacing_scale', 'border_radius',
        'icon_style', 'personality', 'design_keywords'
    ])
    struct_model = groq_model.with_structured_output(ComponentSpec)
    chain = prompt | struct_model
    response = chain.invoke({
        'screens': screens,
        'key_elements': key_elements,
        'colors': colors,
        'typography': typography,
        'spacing_scale': ds.spacing_scale if ds else [],
        'border_radius': ds.border_radius if ds else "",
        'icon_style': ds.icon_style if ds else "",
        'personality': product.personality if product else "",
        'design_keywords': product.design_keywords if product else [],
    })
    time.sleep(4)
    return {'component_spec': response}


def format_component_spec(state: State) -> State:
    cs = state.get('component_spec')
    if not cs:
        return {'component_summary': "No component spec was generated."}

    components = "\n".join(
        f"  - {c.component_name} ({c.component_type}): {c.purpose}\n"
        f"    props: {c.props}\n"
        f"    variants: {[(v.variant_name, v.when_to_use) for v in c.variants]}\n"
        f"    states: {[(s.state_name, s.visual_change) for s in c.states]}\n"
        f"    composition_notes: {c.composition_notes}"
        for c in cs.components
    )
    summary = f"""Naming convention: {cs.naming_convention}
Reuse strategy: {cs.reuse_strategy}
Components:
{components}"""
    return {'component_summary': summary}


def interaction_spec_node(state: State) -> State:
    cs = state.get('component_spec')
    ux = state.get('ux_spec')
    ds = state.get('design_system')
    product = state.get('product_specification')

    components = [c.component_name for c in cs.components] if cs else []
    user_flows = [f.flow_name for f in ux.user_flows] if ux else []

    prompt = PromptTemplate(template=INTERACTION_SPEC_TEMPLATE, input_variables=[
        'components', 'user_flows', 'error_handling_ux', 'empty_states', 'animation_style',
        'personality', 'emotional_goal'
    ])
    struct_model = model.with_structured_output(InteractionSpec)
    chain = prompt | struct_model
    response = chain.invoke({
        'components': components,
        'user_flows': user_flows,
        'error_handling_ux': ux.error_handling_ux if ux else "",
        'empty_states': ux.empty_states if ux else [],
        'animation_style': ds.animation_style if ds else "",
        'personality': product.personality if product else "",
        'emotional_goal': product.emotional_goal if product else "",
    })
    time.sleep(4)
    return {'interaction_spec': response}


def format_interaction_spec(state: State) -> State:
    ins = state.get('interaction_spec')
    if not ins:
        return {'interaction_summary': "No interaction spec was generated."}

    rules = "\n".join(
        f"  - trigger={r.trigger} on {r.element} -> {r.behavior} (feedback: {r.feedback}, timing: {r.timing})"
        for r in ins.micro_interactions
    )
    summary = f"""Micro-interactions:
{rules}
Form validation behavior: {ins.form_validation_behavior}
Loading states: {ins.loading_states}
Error states: {ins.error_states}
Transition style: {ins.transition_style}
Gesture support: {ins.gesture_support}"""
    return {'interaction_summary': summary}


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
design_system (established tokens — stay within this palette/type, do not invent a new one): {design_system_summary}
""",
        input_variables=['query', 'research_context', 'image_description', 'design_system_summary']
    )
    struct_model = model.with_structured_output(DesignDirection)
    chain = prompt | struct_model
    response = chain.invoke({
        'query': state['new_request'],
        'research_context': state['research_context'],
        'image_description': state['image_description'],
        'design_system_summary': state.get('design_system_summary', ''),
    })
    return {'design_direction': response}


def format_design_direction(state: State) -> State:
    dd = state.get('design_direction')

    if not dd:
        return {'design_direction_summary': "No design direction was generated."}

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

    return {'design_direction_summary': summary}


def planner(state: State) -> State:
    prompt = PromptTemplate(
        template="""You are an expert software architect and project planner for an AI code generation system.

Your ONLY job is to analyze the user's request together with everything already established about
this project — requirements, product identity, design system tokens, UX spec, component spec,
interaction spec, an optional research report, an optional reference-image schema, and an optional
design direction — and produce a detailed, structured project plan. You do NOT write any code. You
only plan.

---

INPUTS YOU WILL RECEIVE:

1. user_query: the user's project request.
2. research_report: findings about an external site/product referenced by the user. May be empty.
3. design_schema: exact structural extraction from a reference IMAGE (sections, components, exact
   text, colors, positions). May say "No reference image was provided" — then ignore it.
4. design_direction_summary: the project's creative signature — palette, type pairing, layout
   concept, signature element, copy voice.
5. design_system_summary: the project's TOKEN system — exact color values, typography roles,
   spacing scale, border radius, shadows, icon style, animation style, grid system. This is the
   literal implementation vocabulary every visual file must use.
6. ux_summary: the screens that must exist, navigation between them, user flows, responsive
   behavior, accessibility requirements, empty states, and error-handling pattern.
7. component_summary: the reusable component library (name, props, variants, states, composition)
   that screens must be assembled from — do not invent ad hoc one-off components that duplicate
   something already defined here.
8. interaction_summary: exact micro-interaction behavior, form validation timing, loading/error
   states, and transition style for the components above.

---

HOW TO USE EACH INPUT:

- design_schema (when present) is the SOURCE OF TRUTH for exact visual STRUCTURE from a reference
  image — sections, components, exact text, positions. Every section_id/component it lists MUST
  become explicit, concrete responsibilities in your file plan.
- design_system_summary is the SOURCE OF TRUTH for concrete token VALUES (hex codes, font names,
  spacing numbers, radius, shadows). Any file plan responsibility mentioning a color, font, spacing,
  or radius must reference these exact values/names — never invent new ones.
- design_direction_summary is the SOURCE OF TRUTH for the creative signature layered on top of the
  tokens (which token combination to foreground, the one memorable signature_element, copy_voice).
  It must stay consistent with design_system_summary's literal values.
- ux_summary defines the actual file/page inventory: one file (or clearly split set of files) per
  screen listed, wired together exactly as navigates_to/entry_from describe. Accessibility and
  empty-state requirements from ux_summary must become explicit responsibilities wherever relevant.
- component_summary defines what should be built ONCE and reused — if project structure allows
  (e.g. a components/ directory, or a components.js/css section), plan a dedicated file or clearly
  scoped section per reusable component so screens reference it instead of re-implementing it.
- interaction_summary defines the exact runtime behavior (validation timing, loading/error
  treatment, transitions) that whichever file handles interactivity (e.g. app.js) must implement —
  translate each relevant micro_interaction into an explicit responsibility line.
- research_report only fills in behavioral/functional details none of the above cover. If it
  conflicts with design_schema/design_system_summary/ux_summary on anything structural or visual,
  those win.

---

PRECEDENCE RULE (when inputs conflict):

user_query (explicit user instructions) > design_schema (visual ground truth for structure) >
design_system_summary (token ground truth) > ux_summary (screen/flow ground truth) >
design_direction_summary (creative layer on tokens) > component_summary / interaction_summary
(behavioral detail) > research_report (general reference material).

If user_query explicitly contradicts any of the above (e.g. asks for a dark theme when tokens are
light), follow user_query and note the deviation in that section's responsibilities, but keep
everything else consistent with what wasn't contradicted.

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
12. Do not under-build real requests. A multi-screen product needs one plan entry per screen from
    ux_summary plus a shared tokens/component layer, unless the user explicitly asked for a single
    file output.
13. If design_schema, ux_summary, or component_summary describe structural elements, reflect that
    structure explicitly in the relevant file's responsibilities — don't flatten it into a vague
    "build the homepage" instruction.

---

IMPORTANT BEHAVIOURS:

- The plan you produce is the only instruction the code generator will receive per file. It never
  sees research_report, design_schema, design_system_summary, ux_summary, component_summary, or
  interaction_summary directly — if a detail matters, it must appear in your responsibilities text.
- Always double check that depends_on relationships are consistent — if app.js depends_on
  index.html's element IDs, make that clear in the responsibilities of both files so the code
  generator keeps them in sync.
- Never let research_report, design_schema, design_system_summary, ux_summary, component_summary,
  or interaction_summary override explicit user instructions in user_query.

---

user_query: {query}

research_report: {research_context}

design_schema: {image_description}

design_system_summary: {design_system_summary}

design_direction_summary: {design_direction_summary}

ux_summary: {ux_summary}

component_summary: {component_summary}

interaction_summary: {interaction_summary}""",
        input_variables=[
            'query', 'research_context', 'image_description', 'design_system_summary',
            'design_direction_summary', 'ux_summary', 'component_summary', 'interaction_summary'
        ]
    )
    struct_model = model.with_structured_output(all_files)
    chain = prompt | struct_model
    response = chain.invoke({
        'query': state['new_request'],
        'research_context': state['research_context'],
        'image_description': state['image_description'],
        'design_system_summary': state.get('design_system_summary', ''),
        'design_direction_summary': state.get('design_direction_summary', ''),
        'ux_summary': state.get('ux_summary', ''),
        'component_summary': state.get('component_summary', ''),
        'interaction_summary': state.get('interaction_summary', ''),
    })
    time.sleep(4)
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
    colors, text content, and typography as ground truth for THIS file's visual STRUCTURE
    (positions, sizes, exact text, per-component colors), not as loose inspiration.
16. Reproduce text_content fields EXACTLY character-for-character wherever this file renders
    visible text. Do not paraphrase, shorten, or "improve" the wording.
17. Convert bounding boxes (given as % of image width/height) into responsive layout code —
    percentage widths, flex/grid proportions — never hardcoded pixel positions.
18. Map relative typography sizes to concrete values using this fixed scale: largest=2.5rem,
    large=1.75rem, medium=1.25rem, small=1rem, smallest=0.875rem. Do not invent a different scale.
19. For any component where image_description is filled in, render a placeholder that matches
    the described content and approximate aspect ratio — never fabricate a real image URL.
20. Only use details explicitly present in design_schema. If it doesn't cover something this
    file needs, fall back to design_system/design_direction, then to standard convention.
21. If design_schema is empty, ignore it completely.

---

USING design_system (when provided and non-empty):

22. design_system is the SOURCE OF TRUTH for exact token VALUES — colors, font names/weights/sizes,
    spacing scale, border radius, shadows, icon style, animation timing, grid system. Every raw
    value this file uses must match a token from design_system, never an invented value.
23. If this file defines shared visual tokens (e.g. a CSS file with :root custom properties, a
    Tailwind config, a theme constants file), define design_system's tokens there ONCE, named
    clearly (e.g. --color-accent, --font-display, --space-4), so every other file references those
    names instead of repeating raw values.
24. If this file is NOT the shared-tokens file but still renders visuals, reference the shared
    token names/variables defined by its dependency rather than hardcoding raw values again — check
    the dependency code provided above for what those names actually are.
25. If design_system is empty, fall back to design_direction, then design_schema, then standard
    convention.

---

USING design_direction (when provided and non-empty):

26. design_direction is the creative signature layered on top of design_system's tokens — which
    combination to foreground and the one memorable signature_element. If this file's
    responsibilities call out the signature_element, implement it as the single most deliberate,
    polished piece of this file — spend extra care here specifically (spacing, motion, detail).
27. Wherever this file writes its own visible text (headings, labels, button text, empty states,
    error messages), match copy_voice exactly — specific and active, never generic marketing filler
    like "Welcome to our platform" or "Get started today."
28. design_schema governs exact structural extraction from a reference image; design_direction
    governs creative signature. If design_schema's extracted colors/fonts conflict with
    design_system/design_direction, design_schema wins for that specific component, but stay within
    the established identity for anything design_schema doesn't explicitly specify.
29. If design_direction is empty, ignore it and fall back to design_system/design_schema, then to
    standard, sensible, internally-consistent convention.

---

CRITICAL OUTPUT FORMAT REQUIREMENT:

30. Your response must contain ONLY raw code.
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
design_schema (if applicable): {image_description}
design_system (if applicable): {design_system}
design_direction (if applicable): {design_direction}"""


def code_generator(state: State) -> State:
    """- sorts files by generate_order so dependencies are built first
    - only feeds each file the code of its actual depends_on list (not everything)
    - reads pre-computed summaries directly from state (design_direction_summary,
      design_system_summary) instead of calling the format_* nodes as helper
      functions — those are graph nodes with signature (state: State) -> State now,
      not (SomeSpec) -> str helpers.
    """
    plan = state['fileplans']
    ordered_files = sorted(plan.files, key=lambda f: f.generate_order)

    project_blueprint_ctx = "\n".join([
        f"- File: {f.filename} ({f.language}) [order={f.generate_order}]\n"
        f"  Purpose: {f.purpose}\n"
        f"  Depends on: {f.depends_on}"
        for f in ordered_files
    ])

    design_direction_ctx = state.get('design_direction_summary', "No design direction was generated.")
    design_system_ctx = state.get('design_system_summary', "No design system was generated.")

    code_for_each_file: dict[str, str] = {}

    prompt = PromptTemplate(
        template=CODE_GEN_TEMPLATE,
        input_variables=[
            'blueprint', 'file', 'description', 'responsibilities',
            'depends_on', 'package', 'research_context', 'image_description',
            'design_system', 'design_direction', 'code'
        ]
    )
    chain = prompt | coding_model | StrOutputParser()

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
            'design_system': design_system_ctx,
            'design_direction': design_direction_ctx,
            'code': dep_context,
        })

        code_for_each_file[f.filename] = response
        time.sleep(2)

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
graph.add_node('design_direction', design_direction)
graph.add_node('format_design_direction', format_design_direction)
graph.add_node('requirements', requirements)
graph.add_node('product_spec', product_spec)
graph.add_node('design_system', design_system)
graph.add_node('format_design_system', format_design_system)
graph.add_node('ux_spec', ux_spec_node)
graph.add_node('format_ux_spec', format_ux_spec)
graph.add_node('component_spec', component_spec_node)
graph.add_node('format_component_spec', format_component_spec)
graph.add_node('interaction_spec', interaction_spec_node)
graph.add_node('format_interaction_spec', format_interaction_spec)

graph.add_edge(START, 'query_optimizer')
graph.add_edge('query_optimizer', 'call_researchor_or_not')
graph.add_edge('query_optimizer', 'get_image_schema')
graph.add_edge('get_image_schema', 'format_design_schema')
graph.add_edge('call_researchor_or_not', 'checker')

# requirements waits on both branches so image_description/research_context always exist first
graph.add_edge('format_design_schema', 'requirements')
graph.add_edge('checker', 'requirements')
graph.add_edge('requirements', 'product_spec')
graph.add_edge('product_spec', 'design_system')
graph.add_edge('design_system', 'format_design_system')
graph.add_edge('format_design_system', 'ux_spec')
graph.add_edge('ux_spec', 'format_ux_spec')
graph.add_edge('format_ux_spec', 'component_spec')
graph.add_edge('component_spec', 'format_component_spec')
graph.add_edge('format_component_spec', 'interaction_spec')
graph.add_edge('interaction_spec', 'format_interaction_spec')
graph.add_edge('format_interaction_spec', 'design_direction')
graph.add_edge('design_direction', 'format_design_direction')
graph.add_edge('format_design_direction', 'planner')
graph.add_edge('planner', 'code_generator')
graph.add_edge('code_generator', END)

workflow = graph.compile()


def run_pipeline(prompt: str, reference_image_path: str) -> dict[str, str]:
    """Runs the graph for a given prompt and returns {filename: code}."""
    result = workflow.invoke({'prompt': prompt, 'reference_image_path': reference_image_path})
    return result['file_code']
