---
name: mintlify-docs-writer
description: Use this skill when writing, restructuring, reviewing, or maintaining technical documentation, developer docs, API docs, help-center content, tutorials, how-to guides, explanations, references, navigation, SEO/AEO content, or documentation-quality audits. Do not use it for marketing pages, blog posts, or unrelated prose unless the user explicitly asks to apply documentation best practices.
---

# Mintlify Docs Writer Skill

Use this skill to produce clear, user-centered, maintainable technical documentation based on Mintlify's documentation best practices.

## Core principle

Optimize documentation for the reader's goal. Before drafting or reviewing, determine:

1. Who is the primary audience?
2. What are they trying to accomplish?
3. How much prior knowledge do they have?
4. What content type best fits the goal?
5. What evidence, product facts, code, API details, or screenshots are available?

Avoid writing for every possible audience in one page. Prefer one clear audience and one clear job per page.

## Content-type selection

Choose one primary content type before writing. Do not mix content types unless the user needs a hybrid page.

| Type | Use when the reader wants to | Structure |
| --- | --- | --- |
| Tutorial | Learn through guided practice | Sequential steps, expected outcome, milestones |
| How-to guide | Solve a specific problem | Goal-first, concise steps, minimal background |
| Reference | Look up precise technical facts | Scannable sections, tables, parameters, examples |
| Explanation | Understand a concept or tradeoff | Context, rationale, alternatives, constraints |

When the request is ambiguous, infer the most likely type from the user's goal. Ask only when the content type would materially change the answer.

## Writing rules

Write documentation that is:

- Concise: remove words that do not help the user achieve the goal.
- Clear: prefer simple, direct phrasing over clever or product-centric language.
- Active: tell the user what to do.
- Skimmable: use descriptive headings, short paragraphs, lists, and tables when useful.
- Consistent: use the same term for the same concept across the page.
- Reader-oriented: use "you" where natural.
- Useful: avoid obvious filler such as "Click Save to save."

When editing existing docs, preserve the requested artifact, structure, and scope unless the user asks for a rewrite. Improve clarity, flow, correctness, and consistency without inventing product facts.

## Structure and navigation

For new or reorganized docs:

1. Start with the user's goal and the expected result.
2. State prerequisites before steps.
3. Put common paths before edge cases.
4. Group related concepts together.
5. Use descriptive headings that explain what follows.
6. Make pages self-contained enough to be understood from search or AI retrieval.
7. Use semantic Markdown: proper heading hierarchy, lists for lists, tables for tabular data, and fenced code blocks for code.
8. Add internal links with descriptive anchor text when related docs are known.

Avoid vague headings, overloaded categories, buried essential content, inconsistent terms, and outdated/deprecated pages that can be retrieved by humans or AI agents.

## Code and examples

For developer documentation:

- Provide complete, runnable examples when possible.
- Include required imports, environment variables, setup, permissions, and prerequisites.
- Explain placeholders and expected outputs.
- Include sample requests and responses for APIs when useful.
- Document common errors, edge cases, and recovery steps.
- Do not invent endpoints, parameters, SDK behavior, product capabilities, metrics, roadmap items, or customer outcomes.

If source material is incomplete, use placeholders such as `<API_KEY>` or `[confirm endpoint]`, and explicitly mark assumptions.

## Media

Use screenshots, GIFs, diagrams, and videos only when they add clarity that text alone cannot provide.

When recommending or adding media:

- Prefer screenshots for hard-to-describe UI elements.
- Prefer short GIFs for brief complex workflows or changelog-style demos.
- Prefer video for long workflows or abstract concepts.
- Always include alt text for images.
- Consider maintenance cost when the UI changes often.
- Avoid media that becomes documentation debt.

## SEO and AEO

For docs intended to be discoverable by search engines or AI assistants:

- Use descriptive H1/H2/H3 headings.
- Include relevant keywords naturally; do not keyword-stuff.
- Use short paragraphs and structured lists.
- Add descriptive meta titles and descriptions when requested.
- Use descriptive alt text.
- Make each section self-contained enough for AI retrieval.
- Define terms and acronyms at first use.
- State prerequisites and requirements explicitly.
- Include specific examples, edge cases, and error explanations.

## Maintenance

When auditing or improving existing docs:

1. Identify stale, misleading, duplicated, or low-impact content.
2. Prioritize pages by usage, search demand, support tickets, product changes, and user feedback.
3. Recommend automation where appropriate: linters, metadata checks, stale-page flags, OpenAPI/spec drift checks, and CI documentation checks.
4. Mark deprecated features clearly.
5. Remove or rewrite docs that are so inaccurate they would waste user time.
6. For large messy docs, recommend a structured audit before rewriting.

## Measuring success

Use metrics in context. Do not assume that larger numbers are always better.

Useful signals include:

- Support ticket reduction for documented issues.
- Onboarding completion or time-to-first-success.
- Search queries that return no useful results.
- User ratings and open-ended feedback.
- Usability tests where users try to complete tasks from the docs.
- AI assistant question analytics that reveal gaps.
- Trends over time against a relevant baseline.

## Review checklist

Before finalizing, check:

- Primary audience is clear.
- Content type matches the user's goal.
- Title and headings are descriptive.
- Prerequisites are explicit.
- Steps are complete and in logical order.
- Terminology is consistent.
- Code examples are complete or assumptions are labeled.
- Media is necessary and accessible.
- SEO/AEO basics are covered when relevant.
- Deprecated or outdated content is not left ambiguous.
- The final answer includes citations or a reference list when the user asked for refs.

## Output patterns

For a new doc, use:

```markdown
# [Task-oriented title]

[One-paragraph summary of what the user will accomplish.]

## Prerequisites

- ...

## Steps

1. ...

## Verify the result

...

## Troubleshooting

...

## Next steps

...
```

For a docs review, use:

```markdown
## Summary

[Main recommendation.]

## Issues

| Priority | Issue | Why it matters | Suggested fix |
| --- | --- | --- | --- |

## Revised version

...

## Validation checklist

- ...
```

For an information architecture review, use:

```markdown
## Recommended structure

- ...

## Rationale

...

## Migration notes

...
```

## References

This skill is derived from and summarizes the Mintlify Guides repository and its MIT-licensed documentation best practices.

- Mintlify Guides repository: https://github.com/mintlify/guides
- Introduction: https://raw.githubusercontent.com/mintlify/guides/main/introduction.mdx
- Understand your audience: https://raw.githubusercontent.com/mintlify/guides/main/know-your-audience.mdx
- Content types: https://raw.githubusercontent.com/mintlify/guides/main/content-types.mdx
- Organize navigation: https://raw.githubusercontent.com/mintlify/guides/main/navigation.mdx
- Style and tone: https://raw.githubusercontent.com/mintlify/guides/main/writing-style-tips.mdx
- Using media: https://raw.githubusercontent.com/mintlify/guides/main/media.mdx
- SEO: https://raw.githubusercontent.com/mintlify/guides/main/seo.mdx
- Maintenance: https://raw.githubusercontent.com/mintlify/guides/main/maintenance.mdx
- Tracking success: https://raw.githubusercontent.com/mintlify/guides/main/success.mdx
- License: https://raw.githubusercontent.com/mintlify/guides/main/LICENSE.md
- OpenAI Skills guide: https://developers.openai.com/api/docs/guides/tools-skills
- Codex Agent Skills guide: https://developers.openai.com/codex/skills
