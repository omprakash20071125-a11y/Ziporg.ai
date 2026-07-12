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
import re
import tempfile
from pydantic import BaseModel, Field
from phase2_planner import research
from dotenv import load_dotenv
import uuid
from e2b import Sandbox  # execution sandbox
from playwright.sync_api import sync_playwright  # screenshot capture

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ScreenshotSpec(BaseModel):
    desktop: str
    tablet: str
    mobile: str


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


# --- UX Spec ------------------------------------------------------
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


# --- Component Spec ------------------------------------------------
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


# --- Interaction Spec ----------------------------------------------
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


# --- Execution schema ------------------------------------------------
class ExecutionResult(BaseModel):
    status: Literal["success", "failed"]
    url: str | None
    sandbox_id: str | None
    stack_detected: str
    stderr: str | None


# --- UI Review / Patch / Comparison schemas ---------------------------
class UIIssue(BaseModel):
    issue_id: str = Field(description="short unique id, e.g. 'ISS-1'")
    severity: Literal["critical", "major", "minor"]
    category: str = Field(description="e.g. 'layout', 'color', 'typography', 'spacing', 'responsiveness', 'accessibility', 'content', 'interaction'")
    location: str = Field(description="Where on the page this issue is, e.g. 'hero section', 'nav bar', 'mobile footer'")
    description: str = Field(description="What's wrong, concretely and specifically")
    expected: str = Field(description="What the spec (design_system/design_direction/ux/component/interaction) says it should look like")
    observed: str = Field(description="What the screenshot actually shows")
    affected_viewport: List[str] = Field(description="Which of desktop/tablet/mobile this issue is visible on")


class UIReviewResult(BaseModel):
    quality_score: int = Field(description="0-100 overall quality score against the established spec")
    summary: str = Field(description="Two or three sentence overall assessment of the current build")
    issues: List[UIIssue] = Field(description="Concrete, specific issues found — empty list if none")
    meets_bar: bool = Field(description="True if quality_score/issues are acceptable to ship as-is, false if another patch pass is needed")


class PatchItem(BaseModel):
    filename: str = Field(description="Exact filename to modify, must match an existing generated file")
    change_description: str = Field(description="Concrete, specific instruction describing exactly what to change in this file and why")
    related_issue_ids: List[str] = Field(description="issue_id values from the review this change addresses")
    change_type: Literal["style", "layout", "content", "behavior", "accessibility", "responsive"]


class PatchPlan(BaseModel):
    patches: List[PatchItem] = Field(description="File-level change instructions — no code, just precise instructions")
    reasoning: str = Field(description="Why these specific changes address the reviewed issues, and why nothing else needs to change")


class ComparisonResult(BaseModel):
    improved: bool = Field(description="True if the AFTER screenshots are a genuine visual/functional improvement over BEFORE")
    score_delta: int = Field(description="Estimated change in quality score, new minus old (can be negative)")
    reasoning: str = Field(description="Concrete comparison of what changed between before/after screenshots")
    remaining_concerns: List[str] = Field(description="Anything still not matching spec after this patch pass")


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
    execution_result: ExecutionResult
    retry_count: int
    screenshot_spec: ScreenshotSpec | None
    project_dir: str                          # stable run id/dir reused across patch passes
    review_result: UIReviewResult | None
    patch_plan: PatchPlan | None
    previous_screenshot_spec: ScreenshotSpec | None
    comparison_result: ComparisonResult | None
    ui_review_retry_count: int
    pre_patch_file_code: dict[str, str] | None  # snapshot of file_code taken before a patch pass,
                                                 # so a regression can be rolled back instead of silently kept
    reverted_due_to_regression: bool            # flag set by regression_guard, used for routing
    static_check_findings: List[str]            # NEW: deterministic pre-flight findings fed into review/patch


groq_model = ChatGroq(model="llama-3.3-70b-versatile")

# coding_model: kept on OpenRouter per your requirement, but now env-configurable — see
# CODING_MODEL_NAME below. openai/gpt-oss-20b:free benchmarks competitively for coding among
# free options, but it's still a small, free, rate-limited model — see strip_think_tags()
# and detect_common_bugs() below, and the comments on code_generator/patch_generator, for the
# practical consequences of that choice.
CODING_MODEL_NAME = os.environ.get("CODING_MODEL", "openai/gpt-oss-20b:free")
coding_model = ChatOpenRouter(model=CODING_MODEL_NAME, temperature=0)

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
   - Every color must have a clear, specific usage — not "general use." E.g. "primary CTA
     buttons and active nav state," not "accent color."
   - Avoid the three most common AI-default palettes unless target_users/product_category
     explicitly calls for one: (a) cream background + terracotta/rust accent, (b) near-black +
     single neon accent, (c) generic blue-and-white SaaS palette. Justify your palette choice
     through personality/tone, don't default to convention.
   - Give every token a real, memorable name (e.g. "Ink", "Signal Coral", "Paper") not "color1."
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
- Every InteractionRule's element must be a component_name that already exists in the component
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
# Helper: strip reasoning traces
# ---------------------------------------------------------------------------
def strip_think_tags(text: str) -> str:
    """Some OpenRouter models (reasoning models, and some free-tier models
    under load) prepend a <think>...</think> reasoning block before the actual
    answer. CODE_GEN_TEMPLATE and PATCH_GEN_TEMPLATE both require the response's
    first character to be raw code — an unstripped reasoning block breaks that
    contract and corrupts the generated file. Applied to every coding_model
    response before it's used."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.rstrip())
    return cleaned.strip()


# ---------------------------------------------------------------------------
# NEW: deterministic static-analysis pass over generated code.
#
# The vision-model UI reviewer is unreliable for two specific, very common
# failure modes: (a) literal "undefined"/"null"/"NaN" text rendered on screen
# from a data-binding mismatch, and (b) a <link>/<script> tag in HTML pointing
# at a filename that was never actually generated (so the stylesheet/script
# silently 404s and the page falls back to unstyled browser defaults). Both
# are 100% mechanically detectable from the file_code dict itself — no vision
# call needed, no chance of the model missing it. Findings from this pass are
# force-injected as synthetic critical issues in ui_reviewer, so the patch
# loop cannot ship past them even if the vision reviewer's assessment is
# generous.
# ---------------------------------------------------------------------------
_LINK_HREF_RE = re.compile(r'<link[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE)
_SCRIPT_SRC_RE = re.compile(r'<script[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
_LITERAL_BUG_PATTERNS = [
    ">undefined<", ">null<", ">NaN<",
    "${undefined}", "${null}", "${NaN}",
    "'undefined'undefined", "undefinedundefined",
]


def _is_external_ref(ref: str) -> bool:
    return ref.startswith(("http://", "https://", "//", "data:"))


def detect_common_bugs(file_code: dict[str, str]) -> List[str]:
    findings: List[str] = []
    filenames = set(file_code.keys())
    basenames = {os.path.basename(fn) for fn in filenames}

    for fname, content in file_code.items():
        ext = os.path.splitext(fname)[1].lower()
        if ext in (".html", ".htm"):
            for href in _LINK_HREF_RE.findall(content):
                clean = href.split("?")[0].split("#")[0].lstrip("./")
                if clean and not _is_external_ref(href) and clean not in filenames and os.path.basename(clean) not in basenames:
                    findings.append(
                        f"{fname}: <link href=\"{href}\"> does not match any generated file — "
                        f"this stylesheet will 404 and the page will fall back to unstyled browser defaults."
                    )
            for src in _SCRIPT_SRC_RE.findall(content):
                clean = src.split("?")[0].split("#")[0].lstrip("./")
                if clean and not _is_external_ref(src) and clean not in filenames and os.path.basename(clean) not in basenames:
                    findings.append(
                        f"{fname}: <script src=\"{src}\"> does not match any generated file — "
                        f"this script will fail to load and any JS-driven behavior will be broken."
                    )
        if ext in (".html", ".htm", ".js", ".mjs", ".jsx", ".ts", ".tsx"):
            for bad in _LITERAL_BUG_PATTERNS:
                if bad in content:
                    findings.append(
                        f"{fname}: contains literal \"{bad}\" — almost certainly a data-binding/"
                        f"field-name mismatch that will render as visible broken text (e.g. 'undefined')."
                    )
    return findings


def static_analyzer(state: State) -> State:
    """Runs detect_common_bugs over the just-(re)generated file_code. Placed after
    both code_generator and patch_generator (both feed into writer), so every
    pass — first generation and every patch — gets re-checked before deploy."""
    findings = detect_common_bugs(state.get('file_code', {}))
    if findings:
        print(f"static_analyzer_found: {len(findings)} issue(s)")
        for f in findings:
            print(f"  - {f}")
    else:
        print("static_analyzer_clean")
    return {'static_check_findings': findings}


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
    print('query_optimized')
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
    structured_model = model.with_structured_output(PageSpec)
    result = structured_model.invoke([message])
    print('schema_fetched')
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
    print('front_design_schema_complete')
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
    print('requirements_generated')
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
    print('product_spec_generated')
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
    print('design_system_generated')
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
    print('formated_design_system')
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
    print('ux_generated')
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
    print('format_ux__spec_generated')
    return {'ux_summary': summary}


def component_spec_node(state: State) -> State:
    """FIX: uses `model` (Gemini) instead of groq_model — matches every other spec
    node. ComponentSpec is deeply nested (components -> variants -> states), which
    is exactly where a smaller model's structured-output tool-calling tends to drop
    or malform fields."""
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
    struct_model = model.with_structured_output(ComponentSpec)
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
    print('component_ux__spec_generated')
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
    print('format_component_spec_generated')
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
    print('interaction_spec_node_done')
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
    print('formated_interaction_spec')
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
    print('design_direction_generated')
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
    print('format_design_direction')
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
14. If the project's stack is Python and a runnable server entry point is needed, name it app.py
    when practical — a fallback entry-point detector exists downstream, but app.py keeps behavior
    predictable and avoids relying on that fallback.
15. Every HTML file's <link>/<script> references MUST exactly match a filename this plan actually
    produces elsewhere in `files` — never reference a stylesheet/script filename that isn't itself
    a planned file. This is checked mechanically after generation; a mismatch here is an automatic
    build defect.
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
    print('planner_done')
    return {'fileplans': response}


CODE_GEN_TEMPLATE = """You are an expert software engineer generating one file at a time as part of a larger multi-file project.
This is the already generated code of files this one depends on — if empty, do not consider it: {code}
Analyse the blueprint: {blueprint}
---
PREVIOUS ATTEMPT FAILED (if empty, this is a first attempt — ignore this section entirely):
{previous_error}
If the section above is non-empty, this project was already generated once and failed to run in the
execution sandbox. Treat this as a BUG FIX pass, not a fresh generation:
1. Read the error carefully — it tells you the actual failure (stack trace, missing module, syntax
   error, non-2xx server response, timeout, etc.).
2. If the error clearly implicates THIS file (its filename appears in the traceback, or its
   responsibilities obviously match the failure — e.g. a missing import this file should provide,
   a route this file should define, a syntax error class matching this file's language), you MUST
   fix that specific problem. Do not just regenerate similar code and hope — address the exact
   failure mode described.
3. If the error does NOT implicate this file, generate it normally per the blueprint, but avoid
   reintroducing anything that could plausibly cause the same class of failure (e.g. if the error
   was a missing dependency, double check this file's imports/requires are all satisfied by
   depends_on or a real package; if it was an undefined name/route/element, make sure this file
   doesn't reference anything not defined here or in a dependency).
4. Do not silently drop functionality to avoid the error — fix the actual cause.
---
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
FIELD-CONSISTENCY RULE (data-binding bugs are the single most common defect here):
10a. Before reading/writing any object property (e.g. transaction.category, item.desc),
     verify that exact property name is actually produced somewhere in the dependency code
     provided above. If you cannot find where a field is defined, define it consistently
     yourself in this file rather than guessing a name and hoping it matches — a mismatched
     field name renders as literal "undefined"/"null"/"NaN" text in the browser, which is
     treated as a critical, ship-blocking defect downstream.
10b. Never let a template literal or DOM write produce the literal text "undefined", "null",
     or "NaN" — guard any value that might be missing (e.g. `value ?? ''` / a conditional)
     rather than interpolating it raw.
---
ASSET-LINKING RULE:
10c. If this file is an HTML file, every <link href="..."> and <script src="..."> you write
     MUST exactly match a filename listed in the blueprint's file list above (check the
     "File:" lines). Never reference a stylesheet/script filename that the blueprint does not
     also list as a file being generated — a broken reference here means the page silently
     loses all styling/behavior.
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
    - pulls the previous execution_result's stderr (if this is a retry pass
      after a failed sandbox run) and feeds it into the prompt so the model is
      patching a known bug instead of blindly regenerating from scratch.
    KNOWN LIMITATION (left as-is, flagging for visibility): on a sandbox failure
    this still regenerates every file, not just the one implicated by stderr.
    Splitting that out safely needs error->filename attribution logic first;
    doing it naively risks silently leaving a broken file in place. Fine as a
    v2 improvement, not touched here to avoid destabilizing the retry path.
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
    execution_result = state.get('execution_result')
    previous_error_ctx = ""
    if execution_result and execution_result.status == "failed" and execution_result.stderr:
        previous_error_ctx = (
            f"stack_detected: {execution_result.stack_detected}\n"
            f"stderr:\n{execution_result.stderr}"
        )
    code_for_each_file: dict[str, str] = {}
    prompt = PromptTemplate(
        template=CODE_GEN_TEMPLATE,
        input_variables=[
            'blueprint', 'file', 'description', 'responsibilities',
            'depends_on', 'package', 'research_context', 'image_description',
            'design_system', 'design_direction', 'code', 'previous_error'
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
            'previous_error': previous_error_ctx,
        })
        # Strip any reasoning-trace / stray markdown fence the coding_model might
        # emit before the raw code — CODE_GEN_TEMPLATE rule #30 requires the first
        # character of the file to be actual code, and unstripped output would
        # otherwise corrupt the written file.
        code_for_each_file[f.filename] = strip_think_tags(response)
        print(f'code_generator_done: {f.filename}')
    return {'file_code': code_for_each_file}


# writer node
def writer(state: State) -> State:
    """Writes state['file_code'] to disk. Reuses the same project_dir across
    patch passes (instead of a fresh random dir every call) so the on-disk copy
    of a patched project lands next to its original, pre-patch version."""
    base_dir = os.environ.get(
        "OUTPUT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_projects"),
    )
    runid = state.get('project_dir') or str(uuid.uuid4())
    path = os.path.join(base_dir, runid)
    os.makedirs(path, exist_ok=True)
    for filename, content in state['file_code'].items():
        file_path = os.path.join(path, filename)
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
    return {'project_dir': runid}


# ---------------------------------------------------------------------------
# Execution layer — detect stack, run in an e2b sandbox, wait until ready
# ---------------------------------------------------------------------------
def detect_stack(file_code: dict[str, str]) -> str:
    """Detect stack from what was ACTUALLY generated (file_code), not from what the
    planner intended (fileplans). A planner entry for package.json means nothing if
    code_generator never actually produced that file."""
    filenames = set(file_code.keys())
    if "package.json" in filenames:
        return "node"
    if "requirements.txt" in filenames:
        return "python"
    if any(f.endswith(".py") for f in filenames):
        return "python"  # python files with no requirements.txt — may have zero deps
    return "static"


def detect_python_entrypoint(file_code: dict[str, str]) -> str:
    """FIX: was hardcoded to always run `python3 app.py`. Now prefers app.py if it
    was actually generated, otherwise falls back to the first .py file that exists
    (sorted, deterministic) so a python project doesn't fail purely because the
    planner/coding model named the entry file something else (main.py, server.py)."""
    py_files = sorted(f for f in file_code.keys() if f.endswith(".py"))
    if "app.py" in py_files:
        return "app.py"
    return py_files[0] if py_files else "app.py"


def run_commands_for(stack: str, file_code: dict[str, str]) -> str:
    if stack == "static":
        return "cd /home/user/project && python3 -m http.server 3000"
    if stack == "python":
        entry = detect_python_entrypoint(file_code)
        return f"cd /home/user/project && python3 {entry}"
    if stack == "node":
        return "cd /home/user/project && npm run dev -- --host 0.0.0.0 --port 3000"
    return "cd /home/user/project && python3 -m http.server 3000"


INSTALL_COMMANDS = {
    "static": None,
    "python": "cd /home/user/project && pip install -r requirements.txt",
    "node": "cd /home/user/project && npm install",
}


def wait_for_ready(sandbox: Sandbox, port: int = 3000, timeout: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        check = sandbox.commands.run(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}")
        if check.stdout.strip().startswith(("2", "3")):
            return True
        time.sleep(1)
    return False


def execute_project(state: State) -> State:
    file_code = state['file_code']
    stack = detect_stack(file_code)
    sandbox = Sandbox.create(timeout=1200)  # 20 min — survives the review/patch loop later
    try:
        for filename, content in file_code.items():
            sandbox.files.write(f"/home/user/project/{filename}", content)

        if stack == "static" and "index.html" not in file_code:
            return {'execution_result': ExecutionResult(
                status="failed", url=None, sandbox_id=sandbox.sandbox_id,
                stack_detected=stack,
                stderr="No index.html among generated files — static site has no entry point.",
            )}
        if stack == "node" and "package.json" not in file_code:
            return {'execution_result': ExecutionResult(
                status="failed", url=None, sandbox_id=sandbox.sandbox_id,
                stack_detected=stack,
                stderr="Node stack detected but package.json was not generated.",
            )}

        install_cmd = INSTALL_COMMANDS[stack]
        if stack == "python" and "requirements.txt" not in file_code:
            install_cmd = None
        if install_cmd:
            result = sandbox.commands.run(install_cmd, timeout=180)
            if result.exit_code != 0:
                return {'execution_result': ExecutionResult(
                    status="failed", url=None, sandbox_id=sandbox.sandbox_id,
                    stack_detected=stack, stderr=result.stderr,
                )}

        sandbox.commands.run(run_commands_for(stack, file_code), background=True)
        if not wait_for_ready(sandbox):
            return {'execution_result': ExecutionResult(
                status="failed", url=None, sandbox_id=sandbox.sandbox_id,
                stack_detected=stack, stderr="Server did not become ready in time",
            )}
        print('execution_project_done')
        return {'execution_result': ExecutionResult(
            status="success", url=f"https://{sandbox.get_host(3000)}",
            sandbox_id=sandbox.sandbox_id, stack_detected=stack, stderr=None,
        )}
    except Exception as e:
        return {'execution_result': ExecutionResult(
            status="failed", url=None, sandbox_id=sandbox.sandbox_id,
            stack_detected=stack, stderr=str(e),
        )}


def cleanup_failed_sandbox(state: State) -> State:
    """Kills the sandbox from a failed execution attempt and bumps the retry counter
    before looping back to code_generator for a fresh attempt."""
    result = state.get('execution_result')
    retries = state.get('retry_count', 0)
    if result and result.sandbox_id:
        try:
            sandbox = Sandbox.connect(result.sandbox_id)
            sandbox.kill()
        except Exception:
            pass  # sandbox may already be gone — nothing to clean up
    print(f'killed failed sandbox, retry #{retries + 1}')
    return {'retry_count': retries + 1}


def route_after_execution(state: State) -> str:
    """Conditional edge: success moves on, failure retries up to 2 times, then gives up."""
    if state['execution_result'].status == "success":
        return "success"
    if state.get('retry_count', 0) >= 2:
        return "give_up"
    return "retry"


# ---------------------------------------------------------------------------
# Screenshot node — opens the live sandbox URL and captures desktop / tablet /
# mobile screenshots. This node only CAPTURES; it does not judge the UI.
# The ui_reviewer node compares these against the specs and decides whether
# fixes are needed.
# ---------------------------------------------------------------------------
def capture_screenshots(state: State) -> State:
    result = state.get('execution_result')
    if not result or result.status != "success" or not result.url:
        print('skipping_screenshots_no_live_url')
        return {'screenshot_spec': None}
    url = result.url
    run_id = str(uuid.uuid4())[:8]
    out_dir = tempfile.gettempdir()
    desktop_path = os.path.join(out_dir, f"desktop_{run_id}.png")
    tablet_path = os.path.join(out_dir, f"tablet_{run_id}.png")
    mobile_path = os.path.join(out_dir, f"mobile_{run_id}.png")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(3000)  # let animations/late JS settle
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.wait_for_timeout(500)
            page.screenshot(path=desktop_path, full_page=True)
            page.set_viewport_size({"width": 768, "height": 1024})
            page.reload(timeout=30000)
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(2000)
            page.screenshot(path=tablet_path, full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.reload(timeout=30000)
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(2000)
            page.screenshot(path=mobile_path, full_page=True)
            browser.close()
        new_spec = ScreenshotSpec(desktop=desktop_path, tablet=tablet_path, mobile=mobile_path)
        previous = state.get('screenshot_spec')
        print('screenshots_captured')
        return {'screenshot_spec': new_spec, 'previous_screenshot_spec': previous}
    except Exception as e:
        print(f'screenshot_capture_failed: {e}')
        return {'screenshot_spec': None}


# ---------------------------------------------------------------------------
# UI Reviewer — looks at the live screenshots and checks whether the build
# actually matches the design system / design direction / UX / component /
# interaction specs. Produces a quality score + a concrete issue list.
# ---------------------------------------------------------------------------
def _encode_image(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


UI_REVIEW_INSTRUCTIONS = """You are a meticulous senior UI/UX reviewer auditing a freshly generated,
live website. You are shown its desktop, tablet, and mobile screenshots plus the specs it was
supposed to be built against. Your job is to find every real, concrete gap between the spec and
what was actually shipped — not to nitpick subjective taste.
---
design_system (token ground truth — colors, type, spacing, radius, shadows, icon/animation style):
{design_system_summary}
design_direction (creative signature — palette foregrounding, signature_element, copy_voice):
{design_direction_summary}
ux_spec (screens, navigation, flows, responsive behavior, accessibility, empty states, error handling):
{ux_summary}
component_spec (reusable components, variants, states):
{component_summary}
interaction_spec (micro-interactions, validation, loading/error states, transitions):
{interaction_summary}
original user request (for grounding — what was actually asked for):
{query}
---
INSTRUCTIONS
1. Check color usage, typography, spacing, and border radius against design_system's exact token
   values — flag any visible mismatch (wrong hue, inconsistent spacing, wrong font, etc.).
2. Check whether the signature_element from design_direction is actually present and well executed.
3. Check that copy on the page matches copy_voice (no generic filler if the spec asked for specific,
   active-voice copy).
4. Check that the UX spec's key_elements are actually visible for each screen shown, and that
   responsive_behavior is respected across the desktop/tablet/mobile screenshots (broken layouts,
   overflow, illegible text, elements overlapping = real issues).
5. Check accessibility_requirements as far as visually verifiable (contrast, visible focus states if
   shown, readable text sizes).
6. AUTOMATIC CRITICAL FLAGS — treat each of these as a critical-severity issue the moment you see it,
   with no further judgment call needed:
   - Any visible literal text reading "undefined", "null", or "NaN" anywhere on the page.
   - A page that looks like unstyled default browser HTML — system-font bullet lists, no color
     beyond link-blue, no card/section structure — when a design_system with real tokens exists.
     This means the stylesheet almost certainly failed to load; call it out explicitly as a
     "stylesheet not applied / likely broken asset link" issue, not just a generic styling gap.
7. Only report OTHER issues (beyond #6) that are visually verifiable in the screenshots — do not
   invent problems you cannot see. Do not flag things that are simply a matter of taste if they
   don't contradict the spec.
8. Assign severity: critical = breaks usability, badly contradicts the spec, or matches #6 above;
   major = clearly wrong but page still usable; minor = small polish issue.
9. Give an overall quality_score 0-100 reflecting how close this build is to the spec (100 = spec
   perfectly executed, no issues). Any #6 automatic-critical finding caps quality_score at 50 or
   below, regardless of how good anything else looks.
10. Set meets_bar = true only if quality_score is high (broadly, 85+) AND there are no critical or
    major issues remaining. Otherwise false.
Return ONLY a valid JSON object matching exactly this schema:
{{
  "quality_score": int,
  "summary": string,
  "issues": [{{"issue_id": string, "severity": "critical"|"major"|"minor", "category": string, "location": string, "description": string, "expected": string, "observed": string, "affected_viewport": [string, ...]}}, ...],
  "meets_bar": bool
}}
"""


def ui_reviewer(state: State) -> State:
    spec = state.get('screenshot_spec')
    if not spec:
        result = UIReviewResult(
            quality_score=0,
            summary="No screenshots were available to review (build likely failed before deployment).",
            issues=[],
            meets_bar=False,
        )
        print('ui_review_skipped_no_screenshots')
        return {'review_result': result}
    content = [{"type": "text", "text": UI_REVIEW_INSTRUCTIONS.format(
        design_system_summary=state.get('design_system_summary', ''),
        design_direction_summary=state.get('design_direction_summary', ''),
        ux_summary=state.get('ux_summary', ''),
        component_summary=state.get('component_summary', ''),
        interaction_summary=state.get('interaction_summary', ''),
        query=state.get('new_request', ''),
    )}]
    for label, path in [("DESKTOP (1920x1080)", spec.desktop), ("TABLET (768x1024)", spec.tablet),
                         ("MOBILE (390x844)", spec.mobile)]:
        b64 = _encode_image(path)
        if b64:
            content.append({"type": "text", "text": f"--- {label} screenshot ---"})
            content.append({"type": "image_url", "image_url": f"data:image/png;base64,{b64}"})
    message = HumanMessage(content=content)
    structured_model = model.with_structured_output(UIReviewResult)
    result = structured_model.invoke([message])

    # FIX: merge deterministic static_analyzer findings in as forced critical issues.
    # The vision model can miss a literal "undefined" string or a broken stylesheet
    # link if it doesn't render obviously enough in a screenshot — these findings
    # were confirmed by direct inspection of the generated source, so they cannot be
    # argued away by meets_bar being otherwise high.
    static_findings = state.get('static_check_findings') or []
    if static_findings:
        forced_issues = [
            UIIssue(
                issue_id=f"STATIC-{i + 1}",
                severity="critical",
                category="content" if "literal" in finding else "layout",
                location="site-wide (detected via source inspection, not screenshot)",
                description=finding,
                expected="No literal undefined/null/NaN text; every <link>/<script> reference must "
                         "resolve to an actually-generated file.",
                observed=finding,
                affected_viewport=["desktop", "tablet", "mobile"],
            )
            for i, finding in enumerate(static_findings)
        ]
        result.issues = list(result.issues) + forced_issues
        result.meets_bar = False
        if result.quality_score > 50:
            result.quality_score = 50
        result.summary = (
            result.summary
            + " Additionally, automated source inspection found "
            + f"{len(static_findings)} deterministic defect(s) (broken asset links and/or "
              "literal undefined/null/NaN text) that must be patched."
        )

    print(f'ui_review_done score={result.quality_score} meets_bar={result.meets_bar} issues={len(result.issues)}')
    return {'review_result': result}


def route_after_ui_review(state: State) -> str:
    review = state.get('review_result')
    retries = state.get('ui_review_retry_count', 0)
    if review and review.meets_bar:
        return "done"
    if retries >= MAX_UI_REVIEW_RETRIES:
        return "done"
    if not review or not review.issues:
        return "done"
    return "patch"


# ---------------------------------------------------------------------------
# Patch Planner — turns the review's issue list into concrete, file-level
# change instructions. No code here, only precise instructions.
# ---------------------------------------------------------------------------
PATCH_PLANNER_TEMPLATE = """You are a senior frontend tech lead turning a UI review's issue list into
a precise, minimal set of file-level patch instructions. You do NOT write code — you write exact
instructions the patch generator will follow file-by-file.
---
ISSUES FOUND BY THE REVIEWER:
{issues}
---
FILES THAT EXIST IN THIS PROJECT (only reference these — never invent a new filename):
{files}
design_system (token ground truth): {design_system_summary}
design_direction (creative signature): {design_direction_summary}
---
INSTRUCTIONS
1. For every issue, decide which existing file is responsible for the fix (e.g. a color/spacing
   issue usually belongs in the stylesheet; a missing element or wrong copy usually belongs in the
   markup; a broken interaction belongs in the JS file) — use filename values exactly as listed
   above, never a filename that doesn't exist in the project.
2. Issues whose location says "site-wide (detected via source inspection, not screenshot)" came
   from direct source-code inspection, not the visual review — for a broken <link>/<script>
   reference, patch the HTML file to point at the correct existing filename (or patch the missing
   file's name to match, whichever is the smaller, safer change); for a literal undefined/null/NaN
   string, patch whichever file actually does the data binding to guard the missing value instead
   of interpolating it raw. Treat these with the same priority as any other critical issue.
3. Group multiple issues that touch the same file into one PatchItem's change_description if they're
   related, but keep unrelated fixes to the same file as separate PatchItem entries so the reasoning
   stays traceable.
4. change_description must be a concrete, specific instruction (e.g. "Increase hero section top/bottom
   padding to 64px per the spacing scale" or "Fix button hover state to use the Signal Coral token"),
   never vague ("improve styling").
5. Every PatchItem must list the issue_id(s) it addresses in related_issue_ids.
6. Do not propose a patch for a minor issue if fixing it risks destabilizing something already
   working, unless it's trivial and low-risk.
7. Ignore issues you cannot map to any existing file.
---
OUTPUT FORMAT
Return ONLY a valid JSON object matching exactly this schema:
{{
  "patches": [{{"filename": string, "change_description": string, "related_issue_ids": [string, ...], "change_type": "style"|"layout"|"content"|"behavior"|"accessibility"|"responsive"}}, ...],
  "reasoning": string
}}
"""


def patch_planner(state: State) -> State:
    review = state.get('review_result')
    plan = state.get('fileplans')
    file_code = state.get('file_code', {})
    if not review or not review.issues:
        print('patch_planner_skipped_no_issues')
        return {'patch_plan': PatchPlan(patches=[], reasoning="No issues to patch.")}
    issues_ctx = "\n".join(
        f"[{i.issue_id}] severity={i.severity} category={i.category} location={i.location}\n"
        f"  description: {i.description}\n  expected: {i.expected}\n  observed: {i.observed}\n"
        f"  affected_viewport: {i.affected_viewport}"
        for i in review.issues
    )
    if plan:
        files_ctx = "\n".join(f"- {f.filename} ({f.language}): {f.purpose}" for f in plan.files)
    else:
        files_ctx = "\n".join(f"- {fn}" for fn in file_code.keys())
    prompt = PromptTemplate(template=PATCH_PLANNER_TEMPLATE, input_variables=[
        'issues', 'files', 'design_system_summary', 'design_direction_summary'
    ])
    struct_model = model.with_structured_output(PatchPlan)
    chain = prompt | struct_model
    response = chain.invoke({
        'issues': issues_ctx,
        'files': files_ctx,
        'design_system_summary': state.get('design_system_summary', ''),
        'design_direction_summary': state.get('design_direction_summary', ''),
    })
    print(f'patch_planner_done patches={len(response.patches)}')
    return {'patch_plan': response}


# ---------------------------------------------------------------------------
# Patch Generator — applies ONLY the planned changes to the existing files,
# instead of regenerating the whole project from scratch.
# ---------------------------------------------------------------------------
PATCH_GEN_TEMPLATE = """You are an expert software engineer applying a precise, minimal patch to an
existing file that is already part of a working project. Do NOT rewrite the file from scratch — start
from the existing code and change only what the instructions below require.
---
EXISTING FILE CONTENT ({filename}):
{existing_code}
---
REQUIRED CHANGES (apply all of these, and nothing else):
{changes}
---
design_system (token ground truth — use these exact values for anything you touch): {design_system}
design_direction (creative signature — stay consistent with this if you touch visuals): {design_direction}
---
RULES
1. Preserve everything in the existing file that is not related to the required changes —
   structure, naming, IDs, functions, comments — untouched.
2. Apply every listed change concretely and completely.
3. Do not introduce new dependencies, new files, or references to filenames/IDs that don't already
   exist in this file or its established structure.
4. Any value you introduce (color, spacing, font, radius) must come from design_system/design_direction
   above — never invent a new raw value if a token already covers it.
5. If a change instructs you to fix a broken <link>/<script> reference or a literal
   undefined/null/NaN string, make sure the corrected reference exactly matches a real filename,
   and guard any interpolated value that could be missing (e.g. `value ?? ''`) rather than leaving
   it able to render as literal "undefined"/"null"/"NaN" text again.
6. Your response must contain ONLY the complete, updated raw file content.
   Do NOT wrap it in markdown code fences. Do NOT add commentary. The first character of your
   response must be actual code.
"""


def patch_generator(state: State) -> State:
    patch_plan = state.get('patch_plan')
    file_code = dict(state.get('file_code', {}))
    if not patch_plan or not patch_plan.patches:
        print('patch_generator_skipped_no_patches')
        return {'file_code': file_code}
    # Snapshot file_code BEFORE any patch is applied. screenshot_comparator judges
    # the resulting AFTER screenshots against this BEFORE state; if the patch pass
    # turns out to be a regression, regression_guard uses this snapshot to roll
    # back instead of silently keeping a worse build.
    pre_patch_snapshot = dict(file_code)
    patches_by_file: dict[str, list[PatchItem]] = {}
    for p in patch_plan.patches:
        patches_by_file.setdefault(p.filename, []).append(p)
    prompt = PromptTemplate(
        template=PATCH_GEN_TEMPLATE,
        input_variables=['filename', 'existing_code', 'changes', 'design_system', 'design_direction']
    )
    chain = prompt | coding_model | StrOutputParser()
    for filename, patches in patches_by_file.items():
        existing = file_code.get(filename)
        if existing is None:
            print(f'patch_skipped_missing_file: {filename}')
            continue
        changes_text = "\n".join(
            f"- ({p.change_type}) {p.change_description} [addresses: {', '.join(p.related_issue_ids)}]"
            for p in patches
        )
        response = chain.invoke({
            'filename': filename,
            'existing_code': existing,
            'changes': changes_text,
            'design_system': state.get('design_system_summary', ''),
            'design_direction': state.get('design_direction_summary', ''),
        })
        # Same reasoning-trace / stray-fence stripping as code_generator.
        file_code[filename] = strip_think_tags(response)
        print(f'patched_file: {filename}')
    return {'file_code': file_code, 'pre_patch_file_code': pre_patch_snapshot}


# ---------------------------------------------------------------------------
# Screenshot Comparator — compares BEFORE/AFTER screenshots to judge whether
# the patch pass produced a genuine visual improvement.
# ---------------------------------------------------------------------------
COMPARATOR_INSTRUCTIONS = """You are comparing BEFORE and AFTER screenshots of the same website after
a patch pass intended to fix specific issues. Judge honestly whether the AFTER screenshots are a real
improvement, a regression, or unchanged.
---
Issues the patch pass was supposed to fix:
{issues}
Prior quality_score (before this patch pass): {prev_score}
---
INSTRUCTIONS
1. Compare each BEFORE/AFTER pair (desktop, tablet, mobile) directly.
2. Judge whether the specific issues listed above appear to be resolved in the AFTER screenshots.
3. Flag any regression — anything that looks worse in AFTER than BEFORE, even if unrelated to the
   listed issues.
4. Estimate score_delta as your best-guess change in overall quality score (can be negative if this
   patch pass made things worse).
5. List any concerns that still remain unresolved in remaining_concerns.
Return ONLY a valid JSON object matching exactly this schema:
{{
  "improved": bool,
  "score_delta": int,
  "reasoning": string,
  "remaining_concerns": [string, ...]
}}
"""


def screenshot_comparator(state: State) -> State:
    prev = state.get('previous_screenshot_spec')
    curr = state.get('screenshot_spec')
    retries = state.get('ui_review_retry_count', 0) + 1
    if not prev or not curr:
        print('comparator_skipped_no_previous_screenshots')
        return {
            'comparison_result': ComparisonResult(
                improved=True, score_delta=0,
                reasoning="No previous screenshots were available to compare against.",
                remaining_concerns=[],
            ),
            'ui_review_retry_count': retries,
        }
    review = state.get('review_result')
    issues_ctx = "\n".join(f"[{i.issue_id}] {i.description}" for i in review.issues) if review else "none"
    prev_score = review.quality_score if review else None
    content = [{"type": "text", "text": COMPARATOR_INSTRUCTIONS.format(
        issues=issues_ctx, prev_score=prev_score,
    )}]
    for label, path in [("BEFORE - Desktop", prev.desktop), ("AFTER - Desktop", curr.desktop),
                         ("BEFORE - Tablet", prev.tablet), ("AFTER - Tablet", curr.tablet),
                         ("BEFORE - Mobile", prev.mobile), ("AFTER - Mobile", curr.mobile)]:
        b64 = _encode_image(path)
        if b64:
            content.append({"type": "text", "text": f"--- {label} ---"})
            content.append({"type": "image_url", "image_url": f"data:image/png;base64,{b64}"})
    message = HumanMessage(content=content)
    struct_model = model.with_structured_output(ComparisonResult)
    result = struct_model.invoke([message])

    # A patch pass that fixed a deterministic static-analyzer finding (broken link,
    # literal undefined text) is checked mechanically too — don't rely on the vision
    # model alone to notice a fixed/still-broken asset reference.
    remaining_static = state.get('static_check_findings') or []
    if remaining_static:
        result.improved = False
        result.remaining_concerns = list(result.remaining_concerns) + remaining_static

    print(f'comparator_done improved={result.improved} delta={result.score_delta} retry#{retries}')
    return {'comparison_result': result, 'ui_review_retry_count': retries}


def route_after_screenshots(state: State) -> str:
    """First screenshot pass (no previous) goes straight to review; every
    subsequent pass (after a patch) goes through the comparator first. If this
    screenshot pass is itself the confirmation shot taken right after a
    regression revert+redeploy, skip straight to END instead of re-entering the
    review/patch loop (which just discarded a patch pass — re-reviewing the
    same known-good baseline would either loop pointlessly or re-open already
    considered issues)."""
    if state.get('reverted_due_to_regression'):
        return "end"
    if state.get('previous_screenshot_spec'):
        return "compare"
    return "review"


# ---------------------------------------------------------------------------
# regression_guard.
#
# screenshot_comparator computes comparison_result (improved / score_delta /
# regressions). If it found a regression, this node reverts file_code to the
# pre-patch snapshot.
#
# FIX: previously the "stop" branch went straight to END without re-deploying
# the reverted files — so execution_result.url kept serving the regressed
# code even though file_code (and therefore run_pipeline()'s return value) was
# correctly rolled back. Now a regression routes through writer ->
# execute_project -> capture_screenshots again so the live sandbox URL and the
# returned file_code always agree, then route_after_screenshots sends that
# confirmation pass straight to END (see above) instead of re-entering review.
# ---------------------------------------------------------------------------
def regression_guard(state: State) -> State:
    comp = state.get('comparison_result')
    pre = state.get('pre_patch_file_code')
    if comp and not comp.improved and pre:
        print('regression_detected: reverting file_code to pre-patch snapshot and redeploying')
        return {'file_code': pre, 'reverted_due_to_regression': True}
    return {'reverted_due_to_regression': False}


def route_after_regression_guard(state: State) -> str:
    if state.get('reverted_due_to_regression'):
        return "redeploy_reverted"
    return "review"


MAX_UI_REVIEW_RETRIES = 3


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
graph.add_node('static_analyzer', static_analyzer)  # NEW
graph.add_node('writer', writer)
graph.add_node('execute_project', execute_project)
graph.add_node('cleanup_failed_sandbox', cleanup_failed_sandbox)
graph.add_node('capture_screenshots', capture_screenshots)
graph.add_node('ui_reviewer', ui_reviewer)
graph.add_node('patch_planner', patch_planner)
graph.add_node('patch_generator', patch_generator)
graph.add_node('screenshot_comparator', screenshot_comparator)
graph.add_node('regression_guard', regression_guard)

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
# FIX: run the deterministic static_analyzer immediately after every code-generation
# pass (fresh generation AND every patch pass, since both feed into writer through
# this same node) so its findings are already in state before ui_reviewer runs.
graph.add_edge('code_generator', 'static_analyzer')
graph.add_edge('static_analyzer', 'writer')
graph.add_edge('writer', 'execute_project')
graph.add_conditional_edges(
    'execute_project',
    route_after_execution,
    {
        "success": 'capture_screenshots',
        "retry": 'cleanup_failed_sandbox',
        "give_up": END,
    }
)
graph.add_edge('cleanup_failed_sandbox', 'code_generator')
# capture_screenshots -> (first pass -> ui_reviewer | subsequent pass -> screenshot_comparator | post-revert confirmation -> END)
graph.add_conditional_edges(
    'capture_screenshots',
    route_after_screenshots,
    {
        "review": 'ui_reviewer',
        "compare": 'screenshot_comparator',
        "end": END,
    }
)
graph.add_edge('screenshot_comparator', 'regression_guard')
# FIX: regression_guard -> (regression found -> revert file_code, redeploy through
# static_analyzer/writer/execute_project/capture_screenshots so the live URL matches
# the reverted code, then route_after_screenshots sends it to END | otherwise -> ui_reviewer as before)
graph.add_conditional_edges(
    'regression_guard',
    route_after_regression_guard,
    {
        "redeploy_reverted": 'static_analyzer',
        "review": 'ui_reviewer',
    }
)
# ui_reviewer -> (meets_bar / no issues / retries exhausted -> END | otherwise -> patch loop)
graph.add_conditional_edges(
    'ui_reviewer',
    route_after_ui_review,
    {
        "done": END,
        "patch": 'patch_planner',
    }
)
graph.add_edge('patch_planner', 'patch_generator')
# FIX: patch_generator also flows through static_analyzer before writer, so a patch
# pass is re-checked for broken links / literal undefined text just like a fresh build.
graph.add_edge('patch_generator', 'static_analyzer')

workflow = graph.compile()


def run_pipeline(prompt: str, reference_image_path: str) -> dict[str, str]:
    """Runs the graph for a given prompt and returns {filename: code}."""
    result = workflow.invoke({'prompt': prompt, 'reference_image_path': reference_image_path})
    return result['file_code']


if __name__ == "__main__":
    response = workflow.invoke({'prompt': """Design and build a polished, production-ready single-page web app called Ziporg.ai — a "prompt-to-project" generator with the tagline "Unzip your ideas!"

Brand & visual identity
Primary accent: electric blue #0A68FF ("zip-blue")
Secondary accent: spark yellow #FFD60A (used for loading/processing states)
Success green #28A745, error red #DC3545
Neutrals: near-white canvas #F8F9FA, paper white #FFFFFF, ink #212529, slate #6C757D
Headings in 'Plus Jakarta Sans' (600-700 weight), body text in 'Inter' (400 weight)
Consistent 8px-based spacing scale (8/12/16/24/32/48/64/80px) and 8px border radius throughout
Soft layered shadows (sm/md/lg) and swift 200ms ease-out transitions on all interactive elements
A 12-column responsive fluid grid (collapsing to 8 columns on tablet)

Core flow (two screens, cross-fade transition between them)
Home/prompt screen - centered brand header (logo/title + tagline), a single auto-resizing textarea for the user to describe their project idea, inline validation if submitted empty, and a primary "Generate Project" button.
Project status screen - a pulsing animated progress square, a live status label (Processing -> Complete/Error, color-coded yellow/green/red), a "zipper" progress bar that fills left-to-right with a small tab icon riding along it, and contextual buttons that swap in based on state: Cancel Generation (while processing), Download ZIP (on success), Start New Project (on success or error). Errors show an inline message.

Supporting UI elements
A toast notification system (top-right, auto-dismiss, slide-in/fade-out)
A modal dialog for critical errors, with a "Contact Support" action, focus trap, Escape-to-close, and ARIA roles for accessibility
Skeleton loading blocks (text/block/circle variants) with shimmer animation
Outlined icon style (2px stroke, rounded caps/joins) with color variants for success/error/spark states
Full keyboard/ARIA support: aria-live status updates, aria-busy, focus management, disabled states at 50% opacity

Behavior
Simulate the "generation" as an async process with realistic random delay and an occasional simulated failure, respecting cancellation via an AbortController
Cancelling prompts a confirmation dialog before discarding progress
All state transitions (screen changes, button swaps, status colors) should animate smoothly, no jarring pops

Build this as plain HTML/CSS/JS (ES modules), split into logical files: a base stylesheet (tokens, reset, grid, utilities), a components stylesheet (buttons, textarea, status indicators, toasts, skeletons, zipper animation), a small event-bus/utilities module, a modal module, and a main app script wiring up the two-screen flow.""",
        'reference_image_path': ''})
    print(response.get('execution_result'))
