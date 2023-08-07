import json
import os
import string

import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, AIMessage

from camel_converter import to_snake
import snakemd

from config import get_config

#
# Testing only
#

# load sample RFP if it exists (testing only)
SAMPLE_RFP_PATH = os.path.join(get_config().DATA_DIR, "rfp.txt")
if os.path.exists(SAMPLE_RFP_PATH):
    with open(SAMPLE_RFP_PATH) as f:
        sample_rfp_txt = f.read()
else:
    sample_rfp_txt = ""

#
# Helpers
#

# read the json prompts file
with open(get_config().PROMPTS_FILE) as f:
    prompts = json.load(f)


def parse_numbered_list(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = [line for line in lines if line[0] in string.digits]
    lines = [line[2:].strip() for line in lines]
    return lines


#
# State
#
st.session_state.analysis_complete = False

if "rfp_text" not in st.session_state and sample_rfp_txt:
    st.session_state["rfp_text"] = sample_rfp_txt

if "messages" not in st.session_state:
    st.session_state.messages = []


#
# Streamlit app
#

st.set_page_config(
    page_title="ProposalCrafter™",
    page_icon="📑",
)

if get_config().OPENAI_API_KEY is None:
    st.error(
        "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.",
        icon="🔑",
    )
    st.stop()

#
# UI
#

# Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<div class="header-container">
  <img src="./app/static/hero.png" alt="ProposalCrafter" class="header-image">
  <div class="header-text">
    <h2>ProposalCrafter™</h2>
    <p>Crafting Precision Proposals for Compelling Commitments</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


tab1, tab2 = st.tabs(["Main", "About"])

with tab1:
    with st.form("rfp_analysis_form"):
        rfp_text = st.text_area("RFP text", height=200, key="rfp_text")
        submit = st.form_submit_button("Analyze RFP")
        if rfp_text:
            # setup LangChain
            if "qa_chain" not in st.session_state:
                with st.spinner("🤖 Initializing..."):
                    st.session_state.llm = ChatOpenAI(
                        model=get_config().OPENAI_MODEL,
                        openai_api_key=get_config().OPENAI_API_KEY,
                    )
                    st.session_state.text_splitter = TokenTextSplitter(
                        model_name="gpt-3.5-turbo", chunk_size=1000, chunk_overlap=0
                    )
                    st.session_state.texts = st.session_state.text_splitter.split_text(
                        rfp_text.strip()
                    )
                    st.session_state.docs = [
                        Document(page_content=t) for t in st.session_state.texts
                    ]
                    st.session_state.embeddings = OpenAIEmbeddings(
                        openai_api_key=get_config().OPENAI_API_KEY
                    )
                    st.session_state.vectorstore = Chroma.from_documents(
                        st.session_state.docs, st.session_state.embeddings
                    )
                    st.session_state.retriever = (
                        st.session_state.vectorstore.as_retriever(
                            search_type="similarity", search_kwargs={"k": 5}
                        )
                    )
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type="stuff",
                        retriever=st.session_state.retriever,
                    )

                    doc = snakemd.new_doc()
                    st.session_state.analysis_doc = doc

    if submit and "client_name" not in st.session_state:
        with st.spinner("🤖 Identifying client company..."):
            query = "What is the name of the client? Give your answer as just: <name>"
            client_name = st.session_state.qa_chain.run(query=query)
            st.session_state.client_name = client_name
            st.session_state.analysis_doc.add_heading(
                f"RFP Analysis for {client_name}", level=1
            )
            st.session_state.analysis_doc.add_paragraph("by Fuel Talent AI")

    if "client_name" in st.session_state:
        st.markdown(f"## RFP Analysis for {st.session_state.client_name}")

    if submit and "client_description" not in st.session_state:
        with st.spinner("🤖 Generating client description..."):
            query = "Describe the client company.  What is its main purpose?"
            client_description = st.session_state.qa_chain.run(query=query)
            st.session_state.client_description = client_description
            st.session_state.analysis_doc.add_heading("Client Description", level=2)
            st.session_state.analysis_doc.add_paragraph(client_description)

    if "client_description" in st.session_state:
        st.markdown("## Client Description")
        st.markdown(st.session_state.client_description)

    if submit and "summary" not in st.session_state:
        with st.spinner("🤖 Generating summary..."):
            summary_chain = load_summarize_chain(
                st.session_state["llm"], chain_type="map_reduce"
            )
            summary = summary_chain.run(st.session_state.docs)
            st.session_state.summary = summary
            st.session_state.analysis_doc.add_heading("Proposal Summary", level=2)
            st.session_state.analysis_doc.add_paragraph(summary)

    if "summary" in st.session_state:
        st.markdown("## RFP Summary")
        st.markdown(st.session_state.summary)

    if submit and "proposal_improvements" not in st.session_state:
        with st.spinner("🤖 Identifying proposal improvements..."):
            prompt = PromptTemplate(
                template="""
    <RFP>
    {rfp_text}
    </RFP>
    To ensure that a request for proposal (RFP) contains sufficient details for vendors to create high-quality proposals, the following success criteria can be used:
    1. Clarity: Is the RFP clearly written and easy to understand? Does it provide a clear picture of what the company is looking for and what they expect from the vendor?
    2. Scope: Does the RFP include a detailed description of the project scope and objectives? Does it outline the specific tasks and deliverables that the vendor will be responsible for?
    3. Timeline: Does the RFP include a timeline for the project, with specific dates for milestones and deadlines? Is the timeline realistic and achievable?
    4. Budget: Does the RFP include a budget for the project? Is the budget reasonable and commensurate with the scope of work?
    5. Requirements: Does the RFP outline the specific requirements that vendors must meet in order to be considered? Are the requirements clear and measurable?
    6. Evaluation Criteria: Does the RFP include clear criteria for evaluating vendor proposals? Are the criteria objective and relevant to the project?
    7. Communication: Does the RFP provide clear instructions for how vendors can ask questions or seek clarification during the bidding process? Is there a designated point of contact for vendors to communicate with?
    8. Background Information: Does the RFP provide sufficient background information about the company and its goals, so that vendors can understand the context of the project?
    9. Legal and Contractual Details: Does the RFP include all necessary legal and contractual information, such as non-disclosure agreements, indemnification clauses, and payment terms?
    10. Formatting: Is the RFP well-formatted and visually appealing, with clear section headers and formatting that makes it easy to scan and digest?

    Given the above RFP and this evaluation criteria, list any areas the proposal could be improved.
    Focus specifically on details that are missing or unclear or constraints and conditions that are not specified.
    """,
                input_variables=["rfp_text"],
            )
            improvements_chain = LLMChain(llm=st.session_state["llm"], prompt=prompt)
            proposal_improvements = improvements_chain.run(rfp_text=rfp_text)
            st.session_state.proposal_improvements = proposal_improvements
            st.session_state.analysis_doc.add_heading("Proposal Improvements", level=2)
            st.session_state.analysis_doc.add_paragraph(proposal_improvements)

    if "proposal_improvements" in st.session_state:
        st.markdown("## Proposal Improvements")
        st.markdown(st.session_state.proposal_improvements)

    if submit and "proposal_deliverables" not in st.session_state:
        with st.spinner("🤖 Identifying proposal deliverables..."):
            query = "Generate a numbered list of specific actions being requested (i.e., deliverables) that must be performed to create a proposal.  Do not list actions that will be performed once the vendor is selected (e.g., conducting workshops)."
            proposal_deliverables = st.session_state.qa_chain.run(query=query)
            st.session_state.proposal_deliverables = proposal_deliverables
            st.session_state.analysis_doc.add_heading("Proposal Deliverables", level=2)
            st.session_state.analysis_doc.add_paragraph(proposal_deliverables)

    if "proposal_deliverables" in st.session_state:
        st.markdown("## Proposal Deliverables")
        st.markdown(st.session_state.proposal_deliverables)

    if submit and "post_proposal_actions" not in st.session_state:
        with st.spinner("🤖 Identifying post-proposal actions..."):
            query = "Generate a numbered list of specific actions the vendor will perform if selected (i.e., post-proposal actions)."
            post_proposal_actions = st.session_state.qa_chain.run(query=query)
            st.session_state.post_proposal_actions = post_proposal_actions
            st.session_state.analysis_doc.add_heading("Post-Proposal Actions", level=2)
            st.session_state.analysis_doc.add_paragraph(post_proposal_actions)

    if "post_proposal_actions" in st.session_state:
        st.markdown("## Post-Proposal Actions")
        st.markdown(st.session_state.post_proposal_actions)

    if submit and "technical_requirements" not in st.session_state:
        with st.spinner("🤖 Identifying technical requirements..."):
            query = "What are the technical (software) requirements the client is asking for?"
            technical_requirements = st.session_state.qa_chain.run(query=query)
            st.session_state.technical_requirements = technical_requirements
            st.session_state.analysis_doc.add_heading("Technical Requirements", level=2)
            st.session_state.analysis_doc.add_paragraph(technical_requirements)

    if "technical_requirements" in st.session_state:
        st.markdown("## Technical Requirements")
        st.markdown(st.session_state.technical_requirements)

    if submit and "use_cases_text" not in st.session_state:
        with st.spinner("🤖 Identifying use cases..."):
            query = "What are the client's use cases? If none are specifically mentioned, then recommend a set of potential use case areas. Generate a numbered list of use case titles only."
            use_cases_text = st.session_state.qa_chain.run(query=query)
            st.session_state.use_cases_text = use_cases_text
            st.session_state.analysis_doc.add_heading("Use Cases", level=2)
            st.session_state.analysis_doc.add_paragraph(use_cases_text)

    if "use_cases_text" in st.session_state:
        st.markdown("## Use Cases")
        st.markdown(st.session_state.use_cases_text)

    if submit and "use_case_ideas" not in st.session_state:
        # Generate ideas for each use case
        use_cases = parse_numbered_list(use_cases_text)
        use_case_ideas = st.session_state["use_case_ideas"] = []
        prompt = PromptTemplate(
            template="""
    {client_description}
    {summary}
    General technical requirements: {technical_requirements}
    You are an expert AI consultant preparing a proposal for this client.
    You are focused specifically on the use cases: {use_case}.
    Generate a numbered list of 4 innovative ideas for the use case.
    The first 3 ideas should be creative but practical today.
    The last idea should be a moonshot idea.
    Each suggestion should be tailored to the client's unique business needs and explicitly describe the use of requested technology that incorporates the general technical requirements.
    The moonshot idea should assume no technical or financial limitations and rely on fully autonomous and sentient AI.  This would be the ideal solution if the client had unlimited resources.
    """,
            input_variables=["use_case"],
            partial_variables={
                "client_description": client_description,
                "summary": summary,
                "technical_requirements": technical_requirements,
            },
        )
        idea_chain = LLMChain(llm=st.session_state["llm"], prompt=prompt)

        # Note that the UI is inlined here so that it displays while processing
        st.markdown("### Ideas")
        st.session_state.analysis_doc.add_heading("Ideas", level=3)
        for use_case in use_cases:  # [-1:]:
            with st.spinner(f"💡 Generating ideas for {use_case}"):
                ideas_text = idea_chain.run(use_case=use_case)
                use_case_ideas.append({"use_case": use_case, "ideas_text": ideas_text})
                st.session_state.analysis_doc.add_heading(use_case, level=4)
                st.session_state.analysis_doc.add_paragraph(ideas_text)
            # st.markdown(f"#### {use_case}")
            # st.markdown(ideas_text)

        #
        # Post-analysis processing
        #
        st.session_state.analysis_complete = True
        doc_name = to_snake(st.session_state.client_name)
        # ensure dir exists
        os.makedirs(get_config().DATA_DIR, exist_ok=True)
        doc_path = os.path.join(get_config().DATA_DIR, "analysis_doc.md")
        st.session_state.analysis_doc.dump(doc_name, doc_path)
        texts = st.session_state.text_splitter.split_text(
            str(st.session_state.analysis_doc)
        )
        st.session_state.vectorstore.add_texts(texts)

    if "use_case_ideas" in st.session_state:
        for use_case_idea in st.session_state.use_case_ideas:
            st.markdown(f"#### {use_case_idea['use_case']}")
            st.markdown(use_case_idea["ideas_text"])

    # with st.spinner("🤖 Generating timeline..."):
    #     # TODO
    #     pass

    # with st.spinner("🤖 Generating team..."):
    #     # TODO
    #     pass

    # with st.spinner("🤖 Generating budget..."):
    #     # TODO
    #     pass

    #
    # Sidebar chat
    #

    if "vectorstore" in st.session_state:
        with st.sidebar:
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            conv_qa_chain = ConversationalRetrievalChain.from_llm(
                st.session_state.llm,
                retriever=st.session_state.retriever,
                memory=memory,
            )

            # with st.sidebar:
            st.markdown("## Chat")

            user_input = st.text_input("Your question:")

            # handle user input
            if user_input:
                # insert into front of message history
                st.session_state.messages.insert(0, HumanMessage(content=user_input))
                with st.spinner("🤖 Thinking..."):
                    response = conv_qa_chain.run(user_input)
                st.session_state.messages.insert(1, AIMessage(content=response))

            # display message history
            messages = st.session_state.get("messages", [])
            for i, msg in enumerate(messages):
                if i % 2 == 0:
                    message(msg.content, is_user=True, key=str(i) + "_user")
                else:
                    message(msg.content, is_user=False, key=str(i) + "_ai")

with tab2:
    with open("README.md", "r") as f:
        readme = f.read()
    readme = readme.replace("static/", "./app/static/")
    st.markdown(readme, unsafe_allow_html=True)
