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
import traceback
from pydantic import BaseModel, Field
from phase2_planner import research
from dotenv import load_dotenv
import uuid
from e2b import Sandbox  # execution sandbox
from playwright.sync_api import sync_playwright  # screenshot capture

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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
    language: str = Field(default="", description="Language this file should be written in")
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
    change_type: str = Field(description="e.g. style, layout, content, behavior, accessibility, responsive")


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
    screenshot_error: str | None         # last screenshot-capture exception message, if any
    screenshot_retry_count: int          # retry counter dedicated to capture_screenshots
    project_dir: str                          # stable run id/dir reused across patch passes
    review_result: UIReviewResult | None
    patch_plan: PatchPlan | None
    previous_screenshot_spec: ScreenshotSpec | None
    comparison_result: ComparisonResult | None
    ui_review_retry_count: int
    pre_patch_file_code: dict[str, str] | None  # snapshot of file_code taken before a patch pass,
                                                 # so a regression can be rolled back instead of silently kept
    reverted_due_to_regression: bool            # flag set by regression_guard, used for routing
    static_check_findings: List[str]            # deterministic pre-flight findings fed into review/patch


groq_model = ChatGroq(model="llama-3.3-70b-versatile")

# coding_model: kept on OpenRouter per your requirement, but now env-configurable — see
# CODING_MODEL_NAME below. openai/gpt-oss-20b:free benchmarks competitively for coding among
# free options, but it's still a small, free, rate-limited model — see strip_think_tags()
# and detect_common_bugs() below, and the comments on code_generator/patch_generator, for the
# practical consequences of that choice.
coding_model = ChatOpenRouter(model="poolside/laguna-m.1:free", temperature=0)

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
     buttons and active nav state," not "accent color." Usage strings for text colors should
     literally contain the word "text", and usage strings for background colors should contain
     "background" or "surface" or "page" — this is used downstream for automated contrast checks.
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
     sans-serif." If you choose a Google Font, it MUST actually be loaded via a <link> or
     @import in the generated files downstream — this is checked automatically, so only choose
     a webfont you intend to be genuinely loaded, not just named.
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
- Ensure at least one text/background color pair achieves a WCAG AA contrast ratio of 4.5:1 or
  higher — this is verified automatically downstream, and a failing pair will be flagged as a
  defect and sent back for patching.
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
# Helper: infer a file's language from its extension. Used as a safety net
# when the planner model (language is optional in the fileplan schema — see
# FIX note on the fileplan class above) omits the `language` field for a
# file entry. Kept intentionally small/deterministic rather than another LLM
# call — this only needs to be "good enough" for code_generator's prompt
# context, not authoritative.
# ---------------------------------------------------------------------------
_EXTENSION_LANGUAGE_MAP = {
    ".html": "HTML", ".htm": "HTML",
    ".css": "CSS",
    ".js": "JavaScript", ".mjs": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".py": "Python",
    ".json": "JSON",
    ".md": "Markdown",
    ".sh": "Bash",
    ".yml": "YAML", ".yaml": "YAML",
    ".txt": "Text",
    ".cpp": "C++", ".c": "C", ".h": "C/C++ Header",
    ".java": "Java",
    ".go": "Go",
    ".rb": "Ruby",
    ".sql": "SQL",
}


def infer_language(filename: str, fallback: str = "") -> str:
    ext = os.path.splitext(filename)[1].lower()
    return _EXTENSION_LANGUAGE_MAP.get(ext, fallback or "Text")


# ---------------------------------------------------------------------------
# Shared retry wrapper for structured-output chain.invoke() calls (planner,
# patch_planner, and any future node using groq_model.with_structured_output).
# ---------------------------------------------------------------------------
def invoke_structured_with_retry(chain, payload: dict, node_name: str, max_attempts: int = 3):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            last_err = e
            err_name = type(e).__name__
            is_rate_limit = "RateLimit" in err_name or "429" in str(e) or "rate_limit" in str(e).lower()
            print(f"{node_name}_invoke_failed (attempt {attempt}/{max_attempts}, {err_name}): {e!r}")
            if attempt == max_attempts:
                break
            if is_rate_limit:
                delay = 15 * attempt  # 15s, 30s, ... - give the TPM window room to reset
                print(f"{node_name}_rate_limited: sleeping {delay}s before retry")
                time.sleep(delay)
            # else: schema/tool-call failure - retry immediately, fresh sample
    raise last_err


# ---------------------------------------------------------------------------
# Deterministic static-analysis pass over generated code.
#
# The vision-model UI reviewer is unreliable for several specific, very common
# failure modes. Everything in this section is 100% mechanically detectable
# from the file_code dict (and, where noted, the design_system object) — no
# vision call needed, no chance of the model missing it. Findings are
# force-injected as synthetic critical/major issues in ui_reviewer, so the
# patch loop cannot ship past them even if the vision reviewer's assessment
# is generous.
# ---------------------------------------------------------------------------
_LINK_HREF_RE = re.compile(r'<link[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE)
_SCRIPT_SRC_RE = re.compile(r'<script[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
_LITERAL_BUG_PATTERNS = [
    ">undefined<", ">null<", ">NaN<",
    "${undefined}", "${null}", "${NaN}",
    "'undefined'undefined", "undefinedundefined",
]
_IMG_TAG_RE = re.compile(r'<img\b[^>]*>', re.IGNORECASE)
_ANCHOR_TAG_RE = re.compile(r'<a\b[^>]*>', re.IGNORECASE)
_HREF_VALUE_RE = re.compile(r'href\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
_VIEWPORT_META_RE = re.compile(r'<meta[^>]+name=["\']viewport["\']', re.IGNORECASE)


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

            # NEW: missing alt text on <img> tags — accessibility defect, mechanically detectable.
            for img_tag in _IMG_TAG_RE.findall(content):
                if "alt=" not in img_tag.lower():
                    findings.append(
                        f"{fname}: an <img> tag is missing an alt attribute — "
                        f"screen readers cannot describe it, and it fails basic accessibility requirements. "
                        f"Tag: {img_tag[:80]}"
                    )

            # NEW: dead/empty links — href="" or href="#" with no real destination.
            for a_tag in _ANCHOR_TAG_RE.findall(content):
                m = _HREF_VALUE_RE.search(a_tag)
                if m and m.group(1).strip() in ("", "#"):
                    findings.append(
                        f"{fname}: an <a> tag has an empty/placeholder href (\"{m.group(1)}\") — "
                        f"this is a dead link. Tag: {a_tag[:80]}"
                    )

            # NEW: missing responsive viewport meta tag — silently breaks the mobile
            # responsive_behavior the ux_spec/design pipeline explicitly required.
            if "<head" in content.lower() and not _VIEWPORT_META_RE.search(content):
                findings.append(
                    f"{fname}: no <meta name=\"viewport\"> tag found — mobile responsive layout "
                    f"will not render at the correct scale on real devices."
                )

        if ext in (".html", ".htm", ".js", ".mjs", ".jsx", ".ts", ".tsx"):
            for bad in _LITERAL_BUG_PATTERNS:
                if bad in content:
                    findings.append(
                        f"{fname}: contains literal \"{bad}\" — almost certainly a data-binding/"
                        f"field-name mismatch that will render as visible broken text (e.g. 'undefined')."
                    )
    return findings


# ---------------------------------------------------------------------------
# NEW: WCAG contrast-ratio check, computed directly from design_system.colors
# hex values — no vision call, no ambiguity. Pairs a color whose `usage`
# mentions "text" against every color whose `usage` mentions "background" /
# "surface" / "page", and flags any pair below the WCAG AA 4.5:1 threshold
# for normal text. This is a heuristic pairing (usage strings aren't
# structured data), but it's far more reliable than asking a vision model to
# eyeball contrast from a screenshot.
# ---------------------------------------------------------------------------
def _hex_to_rgb(hex_value: str) -> tuple[int, int, int] | None:
    h = hex_value.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(hex1: str, hex2: str) -> float | None:
    rgb1, rgb2 = _hex_to_rgb(hex1), _hex_to_rgb(hex2)
    if rgb1 is None or rgb2 is None:
        return None
    l1, l2 = _relative_luminance(rgb1), _relative_luminance(rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def check_color_contrast(colors: List["ColorToken"]) -> List[str]:
    if not colors:
        return []
    text_colors = [c for c in colors if "text" in c.usage.lower()]
    bg_colors = [c for c in colors if any(k in c.usage.lower() for k in ("background", "surface", "page"))]
    if not text_colors or not bg_colors:
        return []  # can't pair reliably — don't guess, just skip
    findings: List[str] = []
    for t in text_colors:
        for b in bg_colors:
            ratio = _contrast_ratio(t.value, b.value)
            if ratio is None:
                continue
            if ratio < 4.5:
                findings.append(
                    f"Low contrast: text color '{t.name}' ({t.value}) on background '{b.name}' "
                    f"({b.value}) has a contrast ratio of {ratio:.2f}:1 — WCAG AA requires at "
                    f"least 4.5:1 for normal body text. This pairing will be hard or impossible "
                    f"to read for many users."
                )
    return findings


# ---------------------------------------------------------------------------
# NEW: verify any Google Font actually named in design_system.typography is
# actually loaded somewhere in the generated files (via a Google Fonts
# <link>, an @import, or an @font-face). Prevents the common failure where
# the design system prescribes "Playfair Display" but the HTML never loads
# it, so the browser silently falls back to a system serif and nobody
# notices from the screenshots alone.
# ---------------------------------------------------------------------------
_SYSTEM_FONT_MARKERS = {
    "arial", "helvetica", "system-ui", "sans-serif", "serif", "monospace",
    "-apple-system", "segoe ui", "ui-sans-serif", "ui-serif", "ui-monospace",
    "times new roman", "courier new", "verdana", "tahoma", "georgia",
}


def _looks_like_system_font(font_name: str) -> bool:
    return font_name.strip().lower() in _SYSTEM_FONT_MARKERS


def check_font_loading(design_system: "DesignSystemSpec | None", file_code: dict[str, str]) -> List[str]:
    if not design_system or not design_system.typography:
        return []
    combined = "\n".join(file_code.values()).lower()
    findings: List[str] = []
    seen: set[str] = set()
    for t in design_system.typography:
        primary_font = t.font.split(",")[0].strip().strip("'\"")
        if not primary_font or primary_font.lower() in seen or _looks_like_system_font(primary_font):
            continue
        seen.add(primary_font.lower())
        variants = {
            primary_font.lower(),
            primary_font.lower().replace(" ", "+"),
            primary_font.lower().replace(" ", "-"),
        }
        if not any(v in combined for v in variants):
            findings.append(
                f"design_system typography role '{t.role}' specifies font '{primary_font}', but no "
                f"matching Google Fonts <link>, @import, or @font-face was found in any generated "
                f"file — the page will silently fall back to a default system font instead."
            )
    return findings


# ---------------------------------------------------------------------------
# NEW: verify CSS custom properties defined in :root are actually being
# referenced elsewhere, rather than the same hex value being hardcoded
# repeatedly. Catches the common case where a tokens file/section is
# generated correctly but other files/sections ignore it and hardcode raw
# values, defeating the whole point of the design_system's token discipline.
# ---------------------------------------------------------------------------
_ROOT_VAR_RE = re.compile(r'--([\w-]+)\s*:\s*(#[0-9a-fA-F]{3,8})')
_HEX_COLOR_RE = re.compile(r'#[0-9a-fA-F]{3,8}\b')


def check_css_variable_usage(file_code: dict[str, str]) -> List[str]:
    css_like = {fn: c for fn, c in file_code.items() if fn.endswith((".css", ".html", ".htm"))}
    if not css_like:
        return []
    var_map: dict[str, str] = {}
    for content in css_like.values():
        for var_name, hex_value in _ROOT_VAR_RE.findall(content):
            var_map[hex_value.lower()] = var_name
    if not var_map:
        return []
    findings: List[str] = []
    for fname, content in file_code.items():
        if not fname.endswith((".css", ".html", ".htm")):
            continue
        counts: dict[str, int] = {}
        for match in _HEX_COLOR_RE.findall(content):
            val = match.lower()
            if val in var_map:
                counts[val] = counts.get(val, 0) + 1
        for val, count in counts.items():
            # A single occurrence is likely the :root declaration itself; 3+ raw
            # repeats elsewhere means the token is being bypassed, not reused.
            if count >= 3:
                findings.append(
                    f"{fname}: raw color value {val} (matches design token --{var_map[val]}) is "
                    f"hardcoded {count} times instead of using var(--{var_map[val]}) — this "
                    f"defeats the design system's token consistency and makes future theme "
                    f"changes error-prone."
                )
    return findings


# ---------------------------------------------------------------------------
# NEW: lightweight, dependency-free syntax validation for HTML/CSS. Uses the
# standard-library html.parser to catch unclosed/mismatched tags, and a
# simple brace-balance count for CSS. This is deliberately conservative (not
# a full HTML5 validator) — its job is only to catch the kind of gross
# structural error that silently produces a broken/unstyled page, which
# would otherwise only surface several expensive steps later as a vague
# "layout looks off" finding from the vision-based ui_reviewer.
# ---------------------------------------------------------------------------
def validate_markup_and_css(file_code: dict[str, str]) -> List[str]:
    from html.parser import HTMLParser

    findings: List[str] = []
    void_tags = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    class _StructureChecker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: List[str] = []
            self.errors: List[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in void_tags:
                self.stack.append(tag)

        def handle_startendtag(self, tag, attrs):
            pass  # self-closed tag (e.g. <br/>) — nothing to push

        def handle_endtag(self, tag):
            if tag in void_tags:
                return
            if not self.stack:
                self.errors.append(f"stray closing tag </{tag}> with no open tag")
                return
            if self.stack[-1] == tag:
                self.stack.pop()
                return
            if tag in self.stack:
                # Unwind mismatched tags until we find the real match — this is a
                # best-effort recovery, not a claim about which tag is "correct."
                self.errors.append(f"mismatched closing tag </{tag}> (top of stack was <{self.stack[-1]}>)")
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()
            else:
                self.errors.append(f"closing tag </{tag}> has no corresponding open tag")

    for fname, content in file_code.items():
        if fname.endswith((".html", ".htm")):
            checker = _StructureChecker()
            try:
                checker.feed(content)
            except Exception as e:
                findings.append(f"{fname}: HTML failed to parse — {e}")
                continue
            if checker.errors:
                findings.append(
                    f"{fname}: possible malformed HTML structure — " + "; ".join(checker.errors[:3])
                )
            if checker.stack:
                unclosed = ", ".join(checker.stack[:5])
                findings.append(f"{fname}: {len(checker.stack)} tag(s) left unclosed at end of document: {unclosed}")

        if fname.endswith(".css"):
            opens, closes = content.count("{"), content.count("}")
            if opens != closes:
                findings.append(
                    f"{fname}: unbalanced CSS braces ({opens} '{{' vs {closes} '}}') — "
                    f"this is very likely a syntax error that will break styling for everything "
                    f"after the imbalance."
                )

    return findings


def static_analyzer(state: State) -> State:
    """Runs the full deterministic static-analysis suite over the just-(re)generated
    file_code. Placed after both code_generator (via code_reviewer) and
    patch_generator, so every pass — first generation and every patch — gets
    re-checked before deploy. Combines: broken asset links / literal undefined
    text / missing alt text / dead links / missing viewport meta (all from
    detect_common_bugs), WCAG contrast ratio checks against design_system.colors,
    Google Font load verification against design_system.typography, CSS custom
    property reuse checks, and lightweight HTML/CSS syntax validation."""
    file_code = state.get('file_code', {})
    design_system = state.get('design_system')

    findings = detect_common_bugs(file_code)
    if design_system:
        findings += check_color_contrast(design_system.colors)
        findings += check_font_loading(design_system, file_code)
    findings += check_css_variable_usage(file_code)
    findings += validate_markup_and_css(file_code)

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
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
    ])
    # NOTE: was groq_model — llama-3.3-70b-versatile is text-only and rejects
    # multimodal `content` payloads with a 400 (see groq.BadRequestError).
    # `model` (Gemini) is already used for vision elsewhere (ui_reviewer,
    # screenshot_comparator), so reuse it here instead of adding a second
    # vision-capable model just for this node.
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
    struct_model = groq_model.with_structured_output(RequirementSpec)
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
    struct_model = groq_model.with_structured_output(ProductSpec)
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
    struct_model = groq_model.with_structured_output(DesignSystemSpec)
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
    struct_model = groq_model.with_structured_output(UXSpec)
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
    """Uses `model` (Gemini) instead of groq_model — matches every other spec
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
    # FIX: this was groq_model, contradicting the docstring above (which already
    # explained why that's wrong for this schema) — components: List[ComponentDef]
    # where ComponentDef nests List[ComponentVariant] and List[ComponentStateSpec].
    # Same tool_use_failed failure mode as `planner`'s all_files — see that node's
    # comment for the full explanation.
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
    struct_model = groq_model.with_structured_output(InteractionSpec)
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
    struct_model = groq_model.with_structured_output(DesignDirection)
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
16. Every file entry MUST include its `language` field explicitly (e.g. "HTML", "CSS",
    "JavaScript", "Python") — do not omit it, even when it's obvious from the extension.
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
    # NOTE: was groq_model. all_files is a wide object (6 top-level fields) PLUS
    # a nested files: List[fileplan] where fileplan itself has a list field
    # (depends_on). Groq's llama-3.3-70b-versatile tool-calling repeatedly loses
    # the outer object wrapper on schemas this shape — it emits a bare JSON array
    # (just the files list) instead of the full all_files object, which fails
    # Groq's own argument validation with a 400 tool_use_failed error. This is a
    # model-capability limitation, not a transient fault, so retrying with the
    # same model doesn't help. Gemini (`model`) already handles equally-nested
    # schemas elsewhere in this pipeline (e.g. UIReviewResult, PageSpec), so use
    # it here too instead of groq_model.
    struct_model = model.with_structured_output(all_files)
    chain = prompt | struct_model
    response = invoke_structured_with_retry(chain, {
        'query': state['new_request'],
        'research_context': state['research_context'],
        'image_description': state['image_description'],
        'design_system_summary': state.get('design_system_summary', ''),
        'design_direction_summary': state.get('design_direction_summary', ''),
        'ux_summary': state.get('ux_summary', ''),
        'component_summary': state.get('component_summary', ''),
        'interaction_summary': state.get('interaction_summary', ''),
    }, node_name='planner', max_attempts=3)

    valid_files = [f for f in response.files if f.filename and f.filename.strip()]
    if len(valid_files) != len(response.files):
        print(
            f"planner_dropped_invalid_files: "
            f"{len(response.files) - len(valid_files)} entries had an empty/blank filename"
        )

    for f in valid_files:
        if not f.language or not f.language.strip():
            inferred = infer_language(f.filename, fallback=response.language)
            print(f"planner_backfilled_language: {f.filename} -> {inferred}")
            f.language = inferred

    response.files = valid_files
    print('planner_done')
    return {'fileplans': response}


# ---------------------------------------------------------------------------
# NEW: compact "style manifest" — a short, quick-reference cheat sheet built
# once per run from design_system + design_direction, instead of relying
# purely on the two large formatted summary blocks being re-included verbatim
# in every single per-file code_generator/code_reviewer call. The full
# summaries are still included below it (needed for exact hex/spacing
# fidelity), but this short block goes first so the coding_model's attention
# lands on the handful of details that matter most (primary/accent colors,
# type pairing, spacing base, radius, and — critically — the signature
# element) before it has to wade through the longer reference blocks.
# ---------------------------------------------------------------------------
def build_style_manifest(design_system: "DesignSystemSpec | None", design_direction: "DesignDirection | None") -> str:
    lines: List[str] = []
    if design_system:
        primary_colors = design_system.colors[:4]
        color_bits = ", ".join(f"{c.name}={c.value}" for c in primary_colors)
        type_bits = ", ".join(f"{t.role}={t.font}/{t.weight}/{t.size}" for t in design_system.typography[:3])
        lines.append(f"Key colors: {color_bits}")
        lines.append(f"Key type: {type_bits}")
        lines.append(f"Spacing base: {design_system.spacing_scale}")
        lines.append(f"Radius: {design_system.border_radius}")
    if design_direction:
        lines.append(f"Signature element (must be present and well-executed): {design_direction.signature_element}")
        lines.append(f"Copy voice: {design_direction.copy_voice}")
        lines.append(f"Avoid: {design_direction.avoided_cliches}")
    if not lines:
        return "No design tokens available yet — use standard, sensible conventions."
    return "\n".join(f"- {line}" for line in lines)


CODE_GEN_TEMPLATE = """You are an expert software engineer generating one file at a time as part of a larger multi-file project.
---
CORE RULES (highest priority — read these first, they matter more than anything below):
A. Generate code ONLY for the requested file. Stay strictly within its responsibilities.
B. Never let rendered output show literal "undefined", "null", or "NaN" — guard any value that
   might be missing (e.g. `value ?? ''`) instead of interpolating it raw.
C. Every <link href="..."> / <script src="..."> in an HTML file MUST exactly match a filename
   the blueprint lists as being generated — never reference a file that doesn't exist.
D. Use design_system's exact token values (colors, fonts, spacing, radius) — never invent a new
   raw value when a token already covers it. If this file defines shared CSS custom properties,
   define each token ONCE in :root and have every value elsewhere reference var(--token-name).
E. If this file renders visible text and copy_voice is specified, match it exactly — no generic
   filler like "Welcome to our platform."
F. Your response must be ONLY raw code — no markdown fences, no language name, no commentary. The
   first character must be actual code.
---
style_manifest (quick-reference cheat sheet — see full blocks below for exact values): {style_manifest}
---
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
REFERENCE RULES (detailed — consult as needed; CORE RULES above take precedence in any conflict):
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
ACCESSIBILITY RULE:
10d. Every <img> tag must have a meaningful alt attribute. Every <a> tag must have a real
     destination — never leave href="" or href="#" for a link that's meant to go somewhere;
     use a real anchor id, route, or, for genuine no-op placeholders, a clearly commented
     JS-driven handler instead of a bare "#". Include a <meta name="viewport"
     content="width=device-width, initial-scale=1"> in the <head> of any HTML file that has one.
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
26. If design_system specifies a Google Font (a font name that isn't a system-safe stack), you
    MUST actually load it — add the appropriate <link href="https://fonts.googleapis.com/..."> in
    the HTML <head>, or an @import at the top of the relevant CSS file. Naming a font without
    loading it is a defect that gets caught and sent back for patching.
---
USING design_direction (when provided and non-empty):
27. design_direction is the creative signature layered on top of design_system's tokens — which
    combination to foreground and the one memorable signature_element. If this file's
    responsibilities call out the signature_element, implement it as the single most deliberate,
    polished piece of this file — spend extra care here specifically (spacing, motion, detail).
28. Wherever this file writes its own visible text (headings, labels, button text, empty states,
    error messages), match copy_voice exactly — specific and active, never generic marketing filler
    like "Welcome to our platform" or "Get started today."
29. design_schema governs exact structural extraction from a reference image; design_direction
    governs creative signature. If design_schema's extracted colors/fonts conflict with
    design_system/design_direction, design_schema wins for that specific component, but stay within
    the established identity for anything design_schema doesn't explicitly specify.
30. If design_direction is empty, ignore it and fall back to design_system/design_schema, then to
    standard, sensible, internally-consistent convention.
---
CRITICAL OUTPUT FORMAT REQUIREMENT:
31. Your response must contain ONLY raw code.
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
    - builds a compact style_manifest ONCE per run (not per file) so every
      per-file call gets a short priority cheat-sheet ahead of the full
      reference blocks.
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
    style_manifest = build_style_manifest(state.get('design_system'), state.get('design_direction'))
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
            'design_system', 'design_direction', 'code', 'previous_error', 'style_manifest'
        ]
    )
    chain = prompt | coding_model | StrOutputParser()
    for f in ordered_files:
        if not f.filename or not f.filename.strip():
            print(f"code_generator_skipping_invalid_entry: empty filename (purpose={f.purpose!r})")
            continue
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
            'style_manifest': style_manifest,
        })
        code_for_each_file[f.filename] = strip_think_tags(response)
        print(f'code_generator_done: {f.filename}')
    return {'file_code': code_for_each_file}


# ---------------------------------------------------------------------------
# NEW: code_reviewer — a second pass over every freshly-generated file, run
# once immediately after code_generator and before static_analyzer/execution.
# This is item #1 from the quality review: a dedicated self-check step that
# re-reads each file against its own blueprint responsibilities and design
# tokens, and fixes obvious mismatches BEFORE anything reaches the sandbox —
# catching a meaningful fraction of defects a single generation pass misses,
# without waiting for a full execute -> screenshot -> vision-review cycle to
# find them. If the review call fails for any reason (rate limit, empty
# response), the original file is kept unchanged rather than losing content.
# ---------------------------------------------------------------------------
CODE_REVIEW_TEMPLATE = """You are a meticulous senior code reviewer doing a single self-check pass on
ONE already-generated file, before it ships to the execution sandbox. Fix real, concrete defects.
Do NOT rewrite the file for style preference alone, and do NOT change anything that is already
correct — this is a targeted fix pass, not a rewrite.
---
FILE: {filename}
BLUEPRINT RESPONSIBILITIES THIS FILE MUST SATISFY:
{responsibilities}
CURRENT FILE CONTENT:
{content}
---
style_manifest (quick reference): {style_manifest}
design_system (exact token values this file must respect): {design_system}
design_direction (creative signature — signature_element must not be diluted): {design_direction}
---
CHECK FOR AND FIX, ONLY IF ACTUALLY PRESENT:
1. Any responsibility listed above that the current content does not actually satisfy.
2. Raw color/spacing/font values that don't match a design_system token where a token clearly
   should have been used instead.
3. Any literal "undefined", "null", or "NaN" that could render on screen — guard the value.
4. Missing alt text on <img> tags; empty href="" or href="#" placeholder links.
5. Obvious syntax errors — unclosed HTML tags, unbalanced CSS braces.
6. A Google Font named in design_system/design_direction that this file should be loading (via
   <link> or @import) but isn't.
If the file already correctly satisfies everything above, return it completely UNCHANGED.
---
OUTPUT FORMAT: Return ONLY the complete, corrected raw file content. No markdown fences, no
commentary, no explanation of what you changed. The first character must be actual code."""


def code_reviewer(state: State) -> State:
    plan = state.get('fileplans')
    file_code = dict(state.get('file_code', {}))
    if not plan or not file_code:
        print('code_reviewer_skipped_no_files')
        return {'file_code': file_code}

    plan_by_filename = {f.filename: f for f in plan.files}
    style_manifest = build_style_manifest(state.get('design_system'), state.get('design_direction'))
    design_system_ctx = state.get('design_system_summary', '')
    design_direction_ctx = state.get('design_direction_summary', '')

    prompt = PromptTemplate(
        template=CODE_REVIEW_TEMPLATE,
        input_variables=['filename', 'responsibilities', 'content', 'style_manifest', 'design_system', 'design_direction']
    )
    chain = prompt | coding_model | StrOutputParser()

    reviewed: dict[str, str] = {}
    for filename, content in file_code.items():
        plan_entry = plan_by_filename.get(filename)
        responsibilities = (
            "\n".join(f"- {r}" for r in plan_entry.responsibilities)
            if plan_entry else "No blueprint entry found for this file — review generally for correctness."
        )
        try:
            response = chain.invoke({
                'filename': filename,
                'responsibilities': responsibilities,
                'content': content,
                'style_manifest': style_manifest,
                'design_system': design_system_ctx,
                'design_direction': design_direction_ctx,
            })
            cleaned = strip_think_tags(response)
            if cleaned.strip():
                reviewed[filename] = cleaned
                print(f'code_reviewer_reviewed: {filename}')
            else:
                # Empty response from the review pass — keep the original rather
                # than losing the file's content.
                print(f'code_reviewer_empty_response_keeping_original: {filename}')
                reviewed[filename] = content
        except Exception as e:
            print(f'code_reviewer_failed_keeping_original: {filename} ({e!r})')
            reviewed[filename] = content

    return {'file_code': reviewed}


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
    """Was hardcoded to always run `python3 app.py`. Now prefers app.py if it
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

def wait_for_ready(sandbox, timeout=30):
    start = time.time()

    while time.time() - start < timeout:

        try:
            result = sandbox.commands.run(
                "curl -I http://127.0.0.1:3000"
            )

            if result.exit_code == 0:
                return True

        except Exception:
            # Server isn't ready yet.
            pass

        time.sleep(1)

    return False


def _fail(stack: str, sandbox_id: str | None, stderr: str) -> dict:
   
    print(f"execute_project_failed [{stack}]: {stderr}")
    return {'execution_result': ExecutionResult(
        status="failed", url=None, sandbox_id=sandbox_id,
        stack_detected=stack, stderr=stderr,
    )}


def execute_project(state: State) -> State:
    file_code = state['file_code']
    stack = detect_stack(file_code)

    try:
        sandbox = Sandbox.create(timeout=1200)  # 20 min — survives the review/patch loop later
    except Exception as e:
        print(f"execute_project_sandbox_create_failed [{stack}]: {e!r}")
        return _fail(stack, None, f"Sandbox.create() failed: {e}")

    try:
        for filename, content in file_code.items():
            if not filename or not filename.strip():
                print("execute_project_skipping_invalid_file: empty filename")
                continue
            sandbox.files.write(f"/home/user/project/{filename}", content)

        if stack == "static" and "index.html" not in file_code:
            return _fail(stack, sandbox.sandbox_id,
                         "No index.html among generated files — static site has no entry point.")
        if stack == "node" and "package.json" not in file_code:
            return _fail(stack, sandbox.sandbox_id,
                         "Node stack detected but package.json was not generated.")

        install_cmd = INSTALL_COMMANDS[stack]
        if stack == "python" and "requirements.txt" not in file_code:
            install_cmd = None
        if install_cmd:
            result = sandbox.commands.run(install_cmd, timeout=180)
            if result.exit_code != 0:
                return _fail(stack, sandbox.sandbox_id,
                             f"install failed (exit {result.exit_code}): {result.stderr}")

        run_cmd = run_commands_for(stack, file_code)
        print(f"execute_project_starting_server: {run_cmd}")
        sandbox.commands.run(run_cmd, background=True)
        if not wait_for_ready(sandbox):
            boot_log = ""
            try:
                boot_log = sandbox.commands.run(
                    "curl -s -o /dev/null -w 'last_http_code=%{http_code}\\n' http://localhost:3000; "
                    "ps aux | grep -E 'http.server|node|python3' | grep -v grep",
                    timeout=10,
                ).stdout
            except Exception:
                pass
            return _fail(stack, sandbox.sandbox_id,
                         f"Server did not become ready in time. run_cmd={run_cmd!r} "
                         f"diagnostic={boot_log!r}")
        print('execution_project_done')
        return {'execution_result': ExecutionResult(
            status="success", url=f"https://{sandbox.get_host(3000)}",
            sandbox_id=sandbox.sandbox_id, stack_detected=stack, stderr=None,
        )}
    except Exception as e:
        print(f"execute_project_exception [{stack}]: {e!r}")
        return _fail(stack, sandbox.sandbox_id, str(e))


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
        print(f"execute_project_gave_up_after_retries: {state['execution_result'].stderr}")
        return "give_up"
    return "retry"


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------
def _capture(page, url, path):
    """Open page and save screenshot. Retries page.goto/screenshot failures
    (network hiccups, slow-rendering pages) — NOT browser launch failures,
    which are handled one level up in capture_screenshots."""

    for attempt in range(5):
        try:
            page.goto(
                url,
                wait_until="domcontentloaded",  # networkidle is unreliable on
                timeout=30000,                   # pages with animations/polling
            )

            page.wait_for_load_state("load")
            page.wait_for_timeout(2500)

            page.evaluate("""
            async () => {
                if (document.fonts)
                    await document.fonts.ready;
            }
            """)

            # Trigger lazy-loaded elements
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(500)
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(500)

            page.screenshot(
                path=path,
                full_page=True,
                animations="disabled",
            )

            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise RuntimeError(f"Screenshot file was not written or is empty: {path}")

            return

        except Exception:
            if attempt == 4:
                raise
            page.wait_for_timeout(2000)


def _diagnose_playwright_env():
    """One-time diagnostic dump so a future 'Executable doesn't exist' failure
    is immediately explainable from the logs instead of requiring a fresh
    round of guessing. Cheap, so it's fine to run on every attempt."""
    try:
        from pathlib import Path
        import playwright as _pw

        print("=" * 60)
        print("playwright version:", getattr(_pw, "__version__", "unknown"))
        print("HOME:", os.environ.get("HOME"))
        print("PLAYWRIGHT_BROWSERS_PATH:", os.environ.get("PLAYWRIGHT_BROWSERS_PATH"))

        cache = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or (Path.home() / ".cache" / "ms-playwright"))
        print("Expected browser cache dir:", cache)
        print("Cache dir exists:", cache.exists())
        if cache.exists():
            for entry in cache.iterdir():
                print("  found:", entry)
        print("=" * 60)
    except Exception as diag_err:
        print(f"playwright_diagnostic_failed: {diag_err!r}")


MAX_SCREENSHOT_RETRIES = 2


def capture_screenshots(state: State) -> State:

    result = state.get("execution_result")

    if (
        result is None
        or result.status != "success"
        or not result.url
    ):
        print("skipping_screenshots_no_live_url")
        return {
            "screenshot_spec": None,
            "screenshot_error": None,
        }

    url = result.url
    run_id = uuid.uuid4().hex[:8]
    out_dir = tempfile.gettempdir()

    desktop_path = os.path.join(out_dir, f"desktop_{run_id}.png")
    tablet_path = os.path.join(out_dir, f"tablet_{run_id}.png")
    mobile_path = os.path.join(out_dir, f"mobile_{run_id}.png")

    last_error = None

    for launch_attempt in range(3):
        browser = None
        try:
            _diagnose_playwright_env()

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-software-rasterizer",
                        "--disable-background-networking",
                        "--disable-extensions",
                        "--disable-sync",
                    ],
                )

                try:
                    context = browser.new_context()

                    desktop = context.new_page()
                    desktop.set_viewport_size({"width": 1920, "height": 1080})
                    _capture(desktop, url, desktop_path)
                    desktop.close()

                    tablet = context.new_page()
                    tablet.set_viewport_size({"width": 768, "height": 1024})
                    _capture(tablet, url, tablet_path)
                    tablet.close()

                    mobile = context.new_page()
                    mobile.set_viewport_size({"width": 390, "height": 844})
                    _capture(mobile, url, mobile_path)
                    mobile.close()

                    context.close()
                finally:
                    try:
                        browser.close()
                    except Exception:
                        pass

            print("screenshots_captured")
            return {
                "screenshot_spec": ScreenshotSpec(
                    desktop=desktop_path,
                    tablet=tablet_path,
                    mobile=mobile_path,
                ),
                "screenshot_error": None,
            }

        except Exception as e:
            last_error = str(e)
            print(f"screenshot_capture_failed (launch_attempt={launch_attempt + 1}/3)")
            print(traceback.format_exc())

            non_transient_markers = (
                "Executable doesn't exist",
                "shared libraries",
                "cannot open shared object file",
            )
            if any(m in last_error for m in non_transient_markers):
                print("screenshot_capture_non_transient_env_error — skipping further launch retries")
                break

            time.sleep(2)

    return {
        "screenshot_spec": None,
        "screenshot_error": last_error,
    }


def bump_screenshot_retry(state: State) -> State:
    """Increments the dedicated screenshot retry counter and loops back into
    capture_screenshots. This node exists so a transient/flaky browser launch
    (or a just-fixed environment issue that still occasionally hiccups) gets
    a genuine second and third chance instead of failing the whole review
    step on the first attempt."""
    retries = state.get('screenshot_retry_count', 0)
    print(f'screenshot_capture_retry #{retries + 1} (last error: {state.get("screenshot_error")!r})')
    return {'screenshot_retry_count': retries + 1}


def route_after_capture_screenshots(state: State) -> str:
    """This is the retry path for screenshot capture. If capture_screenshots
    succeeded (screenshot_spec is set), behave exactly as before — first pass
    goes to review, a pass after a patch goes to compare, and a post-revert
    confirmation shot goes straight to END.

    If it FAILED (screenshot_spec is None) and there's a live URL to retry
    against, retry up to MAX_SCREENSHOT_RETRIES times before giving up. Only
    after retries are exhausted does it fall through to ui_reviewer — and at
    that point ui_reviewer/route_after_ui_review below make the failure loud
    and explicit in the logs instead of silently completing as if the review
    passed.
    """
    if state.get('screenshot_spec') is not None:
        if state.get('reverted_due_to_regression'):
            return "end"
        if state.get('previous_screenshot_spec'):
            return "compare"
        return "review"

    result = state.get('execution_result')
    has_live_url = bool(result and result.status == "success" and result.url)
    retries = state.get('screenshot_retry_count', 0)

    if has_live_url and retries < MAX_SCREENSHOT_RETRIES:
        return "retry_screenshot"

    print(
        f"screenshot_capture_gave_up_after_retries "
        f"(retries={retries}, last_error={state.get('screenshot_error')!r})"
    )
    if state.get('reverted_due_to_regression'):
        return "end"
    if state.get('previous_screenshot_spec'):
        return "compare"
    return "review"
    
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
2. Check whether the signature_element from design_direction is actually present and well executed
   — look closely; it is often a small, deliberate detail rather than something that dominates the
   whole page, so do not dismiss it just because it isn't the first thing you notice.
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
        screenshot_error = state.get('screenshot_error')
        retries_used = state.get('screenshot_retry_count', 0)
        summary = (
            "Screenshot capture failed after "
            f"{retries_used + 1} attempt(s) — the UI review could NOT run. "
            "This is an infrastructure/tooling failure (e.g. headless browser "
            "environment), not necessarily a problem with the generated site "
            "itself. Ship status is UNVERIFIED, not confirmed-good."
        )
        if screenshot_error:
            summary += f" Last error: {screenshot_error}"
        result = UIReviewResult(
            quality_score=0,
            summary=summary,
            issues=[],
            meets_bar=False,
        )
        print(f"ui_review_skipped_screenshot_infra_failure: {screenshot_error!r}")
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
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    message = HumanMessage(content=content)
    structured_model = model.with_structured_output(UIReviewResult)

    # NOTE: full pixel-accurate cropping of the signature_element (requiring
    # coordinate estimation and an image library) is not implemented here to
    # keep this dependency-free; instead the prompt above explicitly instructs
    # the reviewer to look closely rather than dismiss a small signature
    # detail — a lighter-weight mitigation for the same underlying problem.
    result = None
    last_err = None
    for attempt in range(2):
        try:
            result = structured_model.invoke([message])
            if result is not None:
                break
            print(f"ui_reviewer_got_none_result (attempt {attempt + 1}/2) — retrying")
        except Exception as e:
            last_err = e
            print(f"ui_reviewer_invoke_exception (attempt {attempt + 1}/2): {e!r}")

    if result is None:
        print(f"ui_reviewer_giving_up_after_retries last_err={last_err!r}")
        result = UIReviewResult(
            quality_score=0,
            summary=(
                "The UI review model returned no usable result (possibly blocked by content "
                f"safety filtering, or a malformed response). Last error: {last_err!r}. "
                "Ship status is UNVERIFIED."
            ),
            issues=[],
            meets_bar=False,
        )

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
                         "resolve to an actually-generated file; colors must meet WCAG contrast; "
                         "named webfonts must actually be loaded.",
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
            + f"{len(static_findings)} deterministic defect(s) (broken asset links, contrast "
              "failures, unloaded fonts, accessibility gaps, and/or literal undefined/null/NaN "
              "text) that must be patched."
        )

    print(f'ui_review_done score={result.quality_score} meets_bar={result.meets_bar} issues={len(result.issues)}')
    return {'review_result': result}


def route_after_ui_review(state: State) -> str:
    review = state.get('review_result')
    retries = state.get('ui_review_retry_count', 0)

    if state.get('screenshot_error') and not state.get('screenshot_spec'):
        print(
            "route_after_ui_review: ENDING due to unresolved screenshot "
            f"infrastructure failure — build was shipped WITHOUT automated "
            f"UI verification. last_error={state.get('screenshot_error')!r}"
        )
        return "done"

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
signature_element (PROTECTED — see rule 8 below): {signature_element}
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
   of interpolating it raw; for a low-contrast color pair, patch the CSS/tokens file to adjust the
   offending color to meet 4.5:1; for an unloaded font, add the missing <link>/@import. Treat these
   with the same priority as any other critical issue.
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
8. PROTECTED SIGNATURE ELEMENT: signature_element above is this product's one memorable visual
   detail. If any patch touches the file/section that implements it, the change_description MUST
   explicitly state that the signature element is being preserved or strengthened, never simplified
   away or removed as a side effect of an unrelated fix.
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
    design_direction = state.get('design_direction')
    signature_element = design_direction.signature_element if design_direction else "Not specified."
    prompt = PromptTemplate(template=PATCH_PLANNER_TEMPLATE, input_variables=[
        'issues', 'files', 'design_system_summary', 'design_direction_summary', 'signature_element'
    ])
    struct_model = groq_model.with_structured_output(PatchPlan)
    chain = prompt | struct_model
    response = invoke_structured_with_retry(chain, {
        'issues': issues_ctx,
        'files': files_ctx,
        'design_system_summary': state.get('design_system_summary', ''),
        'design_direction_summary': state.get('design_direction_summary', ''),
        'signature_element': signature_element,
    }, node_name='patch_planner', max_attempts=3)
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
6. If a change description states the signature element must be preserved or strengthened, do not
   let any other part of this patch simplify, shrink, or remove it.
7. Your response must contain ONLY the complete, updated raw file content.
   Do NOT wrap it in markdown code fences. Do NOT add commentary. The first character of your
   response must be actual code.
"""


def patch_generator(state: State) -> State:
    """FIX (Bug 1): previously returned only {'file_code': ..., 'pre_patch_file_code': ...}
    and NEVER set previous_screenshot_spec. That meant:
      - route_after_capture_screenshots' `if state.get('previous_screenshot_spec')` was
        always False, so screenshot_comparator never actually ran after a patch pass.
      - ui_review_retry_count (only ever incremented inside screenshot_comparator)
        therefore never incremented either.
      - route_after_ui_review's `retries >= MAX_UI_REVIEW_RETRIES` check never tripped,
        so the ui_reviewer -> patch_planner -> patch_generator loop had NO upper bound.
    Fixed by snapshotting the CURRENT screenshot_spec (the "before" shot for this patch
    pass) into previous_screenshot_spec on every return, and resetting
    screenshot_retry_count to 0 so the fresh capture round gets its full retry budget
    rather than inheriting a partially-used counter from the previous round.
    """
    patch_plan = state.get('patch_plan')
    file_code = dict(state.get('file_code', {}))
    if not patch_plan or not patch_plan.patches:
        print('patch_generator_skipped_no_patches')
        return {'file_code': file_code}
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
        file_code[filename] = strip_think_tags(response)
        print(f'patched_file: {filename}')
    return {
        'file_code': file_code,
        'pre_patch_file_code': pre_patch_snapshot,
        'previous_screenshot_spec': state.get('screenshot_spec'),
        'screenshot_retry_count': 0,
    }

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
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    message = HumanMessage(content=content)
    struct_model = model.with_structured_output(ComparisonResult)
    result = struct_model.invoke([message])
    remaining_static = state.get('static_check_findings') or []
    if remaining_static:
        result.improved = False
        result.remaining_concerns = list(result.remaining_concerns) + remaining_static

    print(f'comparator_done improved={result.improved} delta={result.score_delta} retry#{retries}')
    return {'comparison_result': result, 'ui_review_retry_count': retries}

def regression_guard(state: State) -> State:
    """DISABLED (per user request): previously reverted file_code to the
    pre-patch snapshot whenever screenshot_comparator judged a patch pass as
    a regression (improved=False). Kept as a no-op pass-through node so
    comparison_result is still computed and logged for visibility upstream in
    screenshot_comparator — it's just no longer acted upon here."""
    comp = state.get('comparison_result')
    if comp and not comp.improved:
        print(
            f"regression_guard_disabled: comparator flagged this pass as a "
            f"regression (delta={comp.score_delta}), but auto-revert is OFF — "
            f"keeping the patched file_code as-is."
        )
    return {'reverted_due_to_regression': False}


def route_after_regression_guard(state: State) -> str:
    if state.get('reverted_due_to_regression'):
        return "redeploy_reverted"
    return "review"

MAX_UI_REVIEW_RETRIES = 3
graph = StateGraph(State)
graph.add_node('query_optimizer', query_optimizer)
graph.add_node('planner', planner)
graph.add_node('code_generator', code_generator)
graph.add_node('code_reviewer', code_reviewer)
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
graph.add_node('static_analyzer', static_analyzer)
graph.add_node('execute_project', execute_project)
graph.add_node('cleanup_failed_sandbox', cleanup_failed_sandbox)
graph.add_node('capture_screenshots', capture_screenshots)
graph.add_node('bump_screenshot_retry', bump_screenshot_retry)
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
# NEW: code_reviewer runs as a second pass immediately after generation,
# before static_analyzer/execution — see code_reviewer() docstring above.
graph.add_edge('code_generator', 'code_reviewer')
graph.add_edge('code_reviewer', 'static_analyzer')
graph.add_edge('static_analyzer', 'execute_project')
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
graph.add_conditional_edges(
    'capture_screenshots',
    route_after_capture_screenshots,
    {
        "review": 'ui_reviewer',
        "compare": 'screenshot_comparator',
        "end": END,
        "retry_screenshot": 'bump_screenshot_retry',
    }
)
graph.add_edge('bump_screenshot_retry', 'capture_screenshots')
graph.add_edge('screenshot_comparator', 'regression_guard')
graph.add_conditional_edges(
    'regression_guard',
    route_after_regression_guard,
    {
        "redeploy_reverted": 'static_analyzer',
        "review": 'ui_reviewer',
    }
)
graph.add_conditional_edges(
    'ui_reviewer',
    route_after_ui_review,
    {
        "done": END,
        "patch": 'patch_planner',
    }
)
graph.add_edge('patch_planner', 'patch_generator')
# patch_generator flows through static_analyzer before execution, so a patch
# pass is re-checked by the full static-analysis suite just like a fresh
# build. (Not routed through code_reviewer again, to keep patch-loop cost
# and latency down — patch_generator's own prompt already carries the same
# guardrails.)
graph.add_edge('patch_generator', 'static_analyzer')

workflow = graph.compile()


def run_pipeline(prompt: str, reference_image_path: str) -> dict[str, str]:
    """Runs the graph for a given prompt and returns {filename: code}."""
    result = workflow.invoke({'prompt': prompt, 'reference_image_path': reference_image_path})
    return result['file_code']
