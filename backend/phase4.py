You reached the start of the range
Jul 13, 2026 at 2:22 PM
Starting Container
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
query_optimized
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
requirements_generated
INFO:     100.64.0.3:44882 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
product_spec_generated
INFO:     100.64.0.2:60008 - "POST /generate HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
design_system_generated
formated_design_system
INFO:     100.64.0.4:37280 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
ux_generated
format_ux__spec_generated
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
component_ux__spec_generated
format_component_spec_generated
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:groq._base_client:Retrying request to /openai/v1/chat/completions in 7.000000 seconds
INFO:     100.64.0.4:37288 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.4:37288 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:groq._base_client:Retrying request to /openai/v1/chat/completions in 3.000000 seconds
INFO:     100.64.0.2:51622 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
interaction_spec_node_done
formated_interaction_spec
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:groq._base_client:Retrying request to /openai/v1/chat/completions in 10.000000 seconds
INFO:     100.64.0.5:29048 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.6:33114 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.2:50458 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.7:42020 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
design_direction_generated
format_design_direction
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:groq._base_client:Retrying request to /openai/v1/chat/completions in 20.000000 seconds
INFO:     100.64.0.8:24390 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.4:11880 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.9:25946 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.3:23430 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.10:26870 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.11:12014 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 400 Bad Request"
planner_invoke_failed (attempt 1/2): BadRequestError('Error code: 400 - {\'error\': {\'message\': "Failed to call a function. Please adjust your prompt. See \'failed_generation\' for more details.", \'type\': \'invalid_request_error\', \'code\': \'tool_use_failed\', \'failed_generation\': \'<function=all_files>[\\n    {\\n        "filename": "index.html",\\n        "purpose": "Landing page for NovaStack",\\n        "responsibilities": [\\n            "Create a hero section with a large split layout, huge heading, description, primary CTA, secondary CTA, badges, rating, and customer count on the left",\\n            "Create a fully CSS-built futuristic dashboard illustration on the right with floating windows, terminal, analytics, AI assistant, notifications, graphs, activity feed, animated glowing connections, and floating cards",\\n            "Create a trusted companies section",\\n            "Create a feature grid with six feature cards",\\n            "Create an interactive dashboard",\\n            "Create a workflow timeline",\\n            "Create a statistics section",\\n            "Create a pricing section",\\n            "Create a testimonials section",\\n            "Create a FAQ section",\\n            "Create a contact section",\\n            "Create a footer",\\n            "Implement semantic HTML, ARIA labels, keyboard navigation, visible focus states, and high contrast for accessibility",\\n            "Implement responsive design with a maximum width of 1280px and a hamburger menu on smaller screens",\\n            "Implement animations and micro-interactions according to interaction_summary",\\n            "Embed CSS and JavaScript code"\\n        ],\\n        "generate_order": 1,\\n        "language": "HTML",\\n        "package": "",\\n        "depends_on": []\\n    },\\n    {\\n        "filename": "index.html",\\n        "purpose": "Embedded CSS for index.html",\\n        "responsibilities": [\\n            "Create a dark futuristic interface with a floating glass navigation, hero section with large split layout, and various sections for features, dashboard, workflow timeline, statistics, pricing, testimonials, FAQ, contact, and footer",\\n            "Style the hero section with a large split layout, huge heading, description, primary CTA, secondary CTA, badges, rating, and customer count on the left",\\n            "Style the futuristic dashboard illustration on the right with floating windows, terminal, analytics, AI assistant, notifications, graphs, activity feed, animated glowing connections, and floating cards",\\n            "Style the feature grid with six feature cards",\\n            "Style the interactive dashboard",\\n            "Style the workflow timeline",\\n            "Style the statistics section",\\n            "Style the pricing section",\\n            "Style the testimonials section",\\n            "Style the FAQ section",\\n            "Style the contact section",\\n            "Style the footer",\\n            "Implement typography with Space Grotesk and Inter fonts",\\n            "Implement color palette with Nova Dark, Nova Primary, Nova Accent, Nova Neutral, Nova Text, and Nova Success colors",\\n            "Implement spacing scale, border radius, shadows, and icon style according to design_system_summary"\\n        ],\\n        "generate_order": 2,\\n        "language": "CSS",\\n        "package": "",\\n        "depends_on": ["index.html"]\\n    },\\n    {\\n        "filename": "index.html",\\n        "purpose": "Embedded JavaScript for index.html",\\n        "responsibilities": [\\n            "Create a floating glass navigation with a hamburger menu on smaller screens",\\n            "Implement page transitions with 200ms ease-out animation",\\n            "Implement form validation behavior with validation on blur and submit, and errors displayed in a toast notification and inline under the InputField",\\n            "Implement loading states with skeletons for page load, spinners for actions, and progress bars for background fetches",\\n            "Implement error states with red border and inline error message under InputField on validation failure, and toast notifications for product-wide errors",\\n            "Implement micro-interactions according to interaction_summary"\\n        ],\\n        "generate_order": 3,\\n        "language": "JavaScript",\\n        "package": "",\\n        "depends_on": ["index.html"]\\n    }\\n]</function>\'}}')
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:groq._base_client:Retrying request to /openai/v1/chat/completions in 24.000000 seconds
INFO:     100.64.0.10:26224 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.12:57062 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.4:55940 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.12:26062 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.9:57098 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.13:43950 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.12:60252 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:groq._base_client:Retrying request to /openai/v1/chat/completions in 2.000000 seconds
INFO:     100.64.0.2:26360 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
planner_done
INFO:     100.64.0.11:17564 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:     100.64.0.12:56156 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.12:56156 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.12:56156 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.7:50898 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.12:13686 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.10:25720 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.10:25720 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.2:55120 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.12:22602 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.11:39718 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.6:13878 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.2:41058 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.6:28878 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.9:29940 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.9:29940 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
code_generator_done: index.html
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:     100.64.0.10:52042 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.2:17578 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.11:57670 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.4:13102 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.2:43212 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.11:11082 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.14:31586 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.14:31586 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.2:53378 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.10:23226 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.7:48618 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.11:36556 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.6:24232 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
code_generator_done: style.css
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:     100.64.0.11:62044 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.14:10360 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
code_generator_done: script.js
static_analyzer_clean
INFO:httpx:HTTP Request: POST https://api.e2b.app/sandboxes "HTTP/2 201 Created"
INFO:httpx:HTTP Request: POST https://sandbox.e2b.app/files?path=%2Fhome%2Fuser%2Fproject%2Findex.html "HTTP/2 200 OK"
INFO:httpx:HTTP Request: POST https://sandbox.e2b.app/files?path=%2Fhome%2Fuser%2Fproject%2Fstyle.css "HTTP/2 200 OK"
INFO:httpx:HTTP Request: POST https://sandbox.e2b.app/files?path=%2Fhome%2Fuser%2Fproject%2Fscript.js "HTTP/2 200 OK"
execute_project_starting_server: cd /home/user/project && python3 -m http.server 3000
INFO:     100.64.0.15:30992 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
execution_project_done
============================================================
playwright version: unknown
HOME: /root
PLAYWRIGHT_BROWSERS_PATH: /app/.playwright-browsers
Expected browser cache dir: /app/.playwright-browsers
Cache dir exists: True
  found: /app/.playwright-browsers/chromium_headless_shell-1228
  found: /app/.playwright-browsers/.links
  found: /app/.playwright-browsers/chromium-1228
  found: /app/.playwright-browsers/ffmpeg-1011
============================================================
INFO:     100.64.0.7:25998 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
screenshots_captured
INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 503 Service Unavailable"
INFO:google_genai._api_client:Retrying google.genai._api_client.BaseApiClient._request_once in 1.3 seconds as it raised ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}.
INFO:     100.64.0.16:50656 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.7:18870 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.16:50664 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
INFO:     100.64.0.15:18736 - "GET /generate/status/bb400aad-bfac-4715-a5fd-3d69d5b37069 HTTP/1.1" 200 OK
ui_review_done score=65 meets_bar=False issues=10
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 400 Bad Request"
ERROR:ziporg:Pipeline failed [job=bb400aad-bfac-4715-a5fd-3d69d5b37069]: Error code: 400 - {'error': {'message': 'tool call validation failed: parameters for tool PatchPlan did not match schema: errors: [`/patches/5/change_type`: value must be one of "style", "layout", "content", "behavior", "accessibility", "responsive"]', 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '<function=PatchPlan>{"patches": [{"filename": "style.css", "change_description": "Update the hero heading font size to 2.5rem and font weight to 600", "related_issue_ids": ["ISS-1"], "change_type": "style"}, {"filename": "style.css", "change_description": "Set the border radius of all major UI elements to 16px", "related_issue_ids": ["ISS-2"], "change_type": "style"}, {"filename": "index.html", "change_description": "Add the missing elements to the \'Interactive Dashboard\' section, including a sidebar, analytics, area chart, bar chart, code editor, deployment timeline, AI assistant panel, logs, CPU usage, memory usage, and live deployment status", "related_issue_ids": ["ISS-3"], "change_type": "content"}, {"filename": "index.html", "change_description": "Add the missing elements to the contact section, including a contact form, address, phone number, and email", "related_issue_ids": ["ISS-4"], "change_type": "content"}, {"filename": "style.css", "change_description": "Implement the \'floating glass\' aesthetic in the navigation bar", "related_issue_ids": ["ISS-5"], "change_type": "layout"}, {"filename": "style.css", "change_description": "Update the navigation bar items to use the display font Space Grotesk", "related_issue_ids": ["ISS-6"], "change_type": "typography"}, {"filename": "script.js", "change_description": "Add the \'count upward\' animation to the statistics section", "related_issue_ids": ["ISS-7"], "change_type": "behavior"}, {"filename": "style.css", "change_description": "Update the internal padding of buttons to 16px", "related_issue_ids": ["ISS-8"], "change_type": "style"}, {"filename": "style.css", "change_description": "Update the internal padding of input fields to 8px", "related_issue_ids": ["ISS-9"], "change_type": "style"}, {"filename": "style.css", "change_description": "Update the internal padding of cards to 16px", "related_issue_ids": ["ISS-10"], "change_type": "style"}], "reasoning": "These patches address the major and minor issues found in the UI review, ensuring that the hero heading font size and weight are correct, the border radius of major UI elements is consistent, the \'Interactive Dashboard\' and contact sections are complete, the navigation bar has the \'floating glass\' aesthetic, the navigation bar items use the correct font, the statistics section has the \'count upward\' animation, and the internal padding of buttons, input fields, and cards is consistent with the design system."}</function>'}}
Traceback (most recent call last):
  File "/app/main.py", line 207, in run_generation_job
    file_code = run_pipeline(prompt, reference_image_path)
  File "/app/phase4.py", line 2660, in run_pipeline
    result = workflow.invoke({'prompt': prompt, 'reference_image_path': reference_image_path})
  File "/usr/local/lib/python3.13/site-packages/langgraph/pregel/main.py", line 3928, in invoke
    for chunk in self.stream(
                 ~~~~~~~~~~~^
        input,
        ^^^^^^
    ...<11 lines>...
        **kwargs,
        ^^^^^^^^^
    ):
    ^
  File "/usr/local/lib/python3.13/site-packages/langgraph/pregel/main.py", line 2982, in stream
    for _ in runner.tick(
             ~~~~~~~~~~~^
        [t for t in loop.tasks.values() if not t.writes],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
