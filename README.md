# ProposalCrafter™

_Crafting Precision Proposals for Compelling Commitments_

<img src="static/hero.png" width="704" />

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Overview

A critical part of the ProjectCrafter suite, ProposalCrafter is an AI-powered module designed to dissect, analyze, and craft compelling project submissions in response to detailed Requests for Proposals (RFPs) from an expansive array of clients across a variety of sectors.

ProposalCrafter acts as a gatekeeper for business development, developing compelling and data-backed project proposals. Using VisionCrafter for requirement analysis, ResearchCrafter for industry insights, and ExpertCrafter to identify required specializations, it formulates a comprehensive and persuasive proposal. TaskCrafter aids in planning task distribution, while JobCrafter ensures role clarity. BudgetCrafter helps it incorporate a cost-effective budget, and RiskCrafter assists in presenting proactive risk management strategies. Together, these elements create a compelling narrative that underscores the client's needs and the unique capabilities of your organization.

### RFP Analysis: Decoding Client Requirements

Upon introduction of an RFP, ProposalCrafter delves into extensive reading and interpretation. From understanding project constraints and budgetary limitations to gaining insight into the client's organizational needs and intellectual property conditions, ProposalCrafter leaves no stone unturned. It exhaustively mines the RFP for vital information that informs the creation of a comprehensive and intriguing proposal.

### In-Depth Research: Adding Contextual Awareness

ProposalCrafter couples its thorough analysis with a deep dive into the client's industrial arena. Comprehending the nuances of the client's field, competitors, and market trends, it augments its understanding of the project requirements. This extensive research ensures that any proposal drafted aligns seamlessly with the client's reality, adding a layer of precision and relevance to the prospective project plan.

### Crafting the Proposal: Translating Insights into Plans

Equipped with the invaluable insights gleaned from its extensive analysis and research, ProposalCrafter enters its core phase. It meticulously crafts a tailored proposal that outlines the project capacity, detailed staffing plans, cost structure, and timeline. Each aspect of the proposal is carefully designed to meet, and exceed, the terms outlined in the RFP, resulting in a compelling project offer that not only satisfies the client's needs but also showcases the unique capabilities of the proposing firm.

### Polished Presentation: Packaging for Persuasion

Once the proposal has been created, ProposalCrafter doesn’t stop there. The AI system encapsulates the well-detailed plan into a polished, attractive PDF presentation. This expert design ensures that every segment of the proposal, from skillset showcase to cost calculation, is presented in the best light, making it easily consumable and highly compelling.

### Summary

In essence, ProposalCrafter is instrumental in crafting precision-targeted, persuasive proposals that respond to client requirements while demonstrating organizational skills and capabilities. It amplifies the effectiveness of the project suite by ensuring that every project submission is high-quality, client-specific, and value-communicating.

## Setup

### Configuration

Create a copy of `.env.template` named `.env` and fill in the required values (e.g., `OPENAI_API_KEY`).

### Dependencies

Create and activate a fresh Python virtual environment, then run:

```bash
pip install -r requirements.txt
```

#### Patching SQLITE3

If you receive the following error:

```bash
RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
Please visit https://docs.trychroma.com/troubleshooting#sqlite to learn how to upgrade.
```

You may need to [patch](https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300?permalink_comment_id=4650539#gistcomment-4650539) the `chromadb` package, e.g., `/home/<user>/.pyenv/versions/3.11.4/envs/gai-proposal-crafter/lib/python3.11/site-packages/chromadb/__init__.py`:

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### Running

Start the [streamlit](https://streamlit.io/) auto-reload server:

```bash
streamlit run app.py
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
