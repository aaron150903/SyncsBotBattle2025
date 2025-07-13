# Internal Styling and Organisation Guide
> Please read through this document before you pushing code to this repository

## Introduction
_Why is styling necessary?_
The aim of this guide is to reduce the range of interpretation of code. Code after is a language. It is particularly important for those who register for bot battle to see uniform style. Like how a grammatical mistake can obfuscate the meaning of a sentence, poor code style makes code more difficult to follow.

## Python Style Guide
We will use a linter and formatter (See [`ruff`](https://docs.astral.sh/ruff/)) as part of our dev environment given to you (See [lin](https://toxigon.com/linters-vs-formatters-explained) for more info). This is automatically enforced with python's `pre-commit` library.

I recommend using this formatting, and linting however they are entirely optional. Have a look at the base readme to get started.

## Bash Style Guide
No bash this year! Yay!

## Git Organisation

Read the guidelines below to write good commit messages, branch names, and make pull requests that follow the conventions we will be using throughout the project.

### Issues
We welcome contribution to this project. Changes to this repository will be received on a issue assigned basis. To contribute...

- [ ] Find an issue that is unassigned, unmarked, or contains the help wanted label.
- [ ] Change code and create a pull request closing the issue - see [link](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue). After which a maintainer will assign you to the issue (If this issue is large in complexity and you want to be assigned to the issue, comment on the issue).

### Commit Messages

- Capitalise the subject line.
- Do not end with a period.
- Use imperative mood, i.e. instead of *"Added ..."* write *"Add ..."*.
- Keep messages logical and relevant, do not write things like *"Please work"* or *"I hate frontend"*. To help decide the extent of this, imagine trying to access a point in the project 2 weeks ago, it would be better to have something like *"Add CSS for Navbar template"* or , so that we know from a glance what the commit is for.
- For more detailed messages, use `git commit -m <title> -m <description>`, however short and concise is still preferred.

### Branch Names
Make a branch using `git checkout -b <branch_name>`.
- Names fall under one of **4** categories
	- Minor Feature: `minor-featureName`
	- Major Feature: `major-featureName`
	- Patch: `patch-patchName`
	- Miscellaneous: `name`
		- For example `documentation` for changing the README, or adding another markdown

_patch: a non API breaking change to the codebase - anyone that uses your code will not need to be concerned with the changes you pushed._
_minor: a minor API change to the codebase - anyone that uses your code will need to slightly modify implementation of their code._
_major: a major API change to the codebase - generally breaks code, wherever used, and will need significant modification._

Examples
- Adding comments - patch
- Image sizing changes (doesn't effect other code) - patch
- Changing div css - minor
- Changing div ordering - minor
- Changing server API endpoints - major as it breaks other code.

### Pull Requests
*Summarised from [this article](https://namingconvention.org/git/pull-request-naming.html).*

#### Title
- Short and descriptive summary
- Start with corresponding ticket/story id (e.g. from Jira, GitHub issue)
- Should be capitalized and written in imperative present tense
- Do not end with period
- Suggested format: *#<Ticket_ID> PR description*

#### Description
- Closing keyword with issue number - see [link](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) (e.g. Closes #0).
- Separated with a blank line from the subject
- Explain changes and justify
- Separate different issues into different paragraphs (capitalising each paragraph)
- We recommend using screenshots over long descriptions (A simple before and after will do)
- If the description is longer than a paragraph include a `TLDR:` one-liner as the first line

#### Example Pull Request
```
TLDR: (Necessary for longer PRs) this PR defines PR message syntax.
resolves/closes/fixes #ISSUE_NO (, resolves/closes/fixes #ISSUE_NO2 (, ...))

This pull request is part of the work to make it easier for people to contribute to naming convention guides.

To achieve this, we have:
- Found an issue
- Made a PR
- Made clear the changes introduced
- Included images
```
