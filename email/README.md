# Intellicamp email: images to export

Export each layer from the Figma file "Recrutly Email template" as **PNG at 2x**, using exactly these file names.
Upload them into the `email/` folder of this repo (Add file → Upload files on GitHub). The template already points to:

    https://raw.githubusercontent.com/tbsdeveloper04-ui/intellicamp/main/email/

If you host them somewhere else, find-and-replace that address in `intellicamp-email.html`.

| File name            | Figma layer (node id)                                   | Shown at |
|----------------------|---------------------------------------------------------|----------|
| hero-bg.png          | Frame 50126557 (30:226): hero gradient + person + cards | 600×632  |
| logo.png             | Intellicamp. full logo (30:219)                         | 184×46   |
| sparkle.png          | Group 1000002545 (37:38): sparkle next to "Power of AI" | 24×26    |
| icon-learn.png       | Frame 50126589 (30:62): Learn                           | 53×53    |
| icon-practice.png    | Frame 50126589 (37:58): Practice                        | 53×53    |
| icon-evaluate.png    | Frame 50126589 (37:71): Evaluate                        | 53×53    |
| icon-analyze.png     | Frame 50126589 (37:84): Analyze                         | 53×53    |
| icon-improve.png     | Frame 50126589 (37:97): Improve                         | 53×53    |
| icon-paper.png       | Frame 50126589 (38:264): "What do you get" card 1       | 53×53    |
| icon-evaluation.png  | Frame 50126589 (38:278): card 2                         | 53×53    |
| icon-insights.png    | Frame 50126589 (38:305): card 3                         | 53×53    |
| icon-reports.png     | Frame 50126589 (38:322): card 4                         | 53×53    |
| workflow.png         | Frame 50126578 (30:144): six-step mockup                | 258×208  |
| chapter-map.png      | image 182 (39:452): chapter map                         | 508×138  |
| cta-bg.png           | 6e790fbd…1 (30:204): free-trial section background      | 600×260  |

Tip: in Figma, select a layer, then Export → "+" → 2x → PNG, and rename the layer to the file name first
so the export comes out named correctly.

## Before sending
- Replace `{{company_address}}` and `{{unsubscribe_url}}` with your sending tool's merge tags
  (in Resend broadcasts the unsubscribe tag is `{{{RESEND_UNSUBSCRIBE_URL}}}`).
- Point the button links at your real sign-up / demo pages (all currently go to intellicamp.in with UTM tags).
