"""
Startup Evaluation Pipeline

This script uses a multi-agent architecture powered by OpenAI's GPT-4o-mini to evaluate startup submissions.
Key features:
- Input/output guardrails to detect PII, hallucinations, bias, and sensitive content
- Specialized agents for evaluating product, team, market, and vision
- Aggregated report synthesis with scoring (0–5 scale)

Usage:
1. Create and activate a virtual environment
2. Install dependencies with `pip install -r requirements.txt`
3. Set your OpenAI API key in `.env` as: `OPENAI_API_KEY=your_key_here`
4. Optionally, define the model version using `MODEL=model_version`
"""



"""
!-----------------------THINGS TO DO-----------------------!
3. Reflection/Critque agent that can provide a quick analysis of the output and review it.
    This can be a good place to use OPENAI SDK vector store as a benchmark place to get the metrics from.
    Metrics can include: Relevancy, Completeness, efficiency. Needs to be short, with the output being at the very end or at the very start.
4. Responses can be more dynamic. Further interaction between user and the orchestrator_agent. Q&A. 
5. Moduleization of the code. Multiple files for easy readability. 
6. Rewrite the Startup Evaluation Pipeline intro section.
"""




"""
----------------------------------------------------Key Dependeices Section----------------------------------------------------
"""
from agents import Agent, Runner, guardrail, handoff, GuardrailFunctionOutput, RunContextWrapper, input_guardrail, output_guardrail, TResponseInputItem, OutputGuardrailTripwireTriggered
import os
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
from pydantic import BaseModel
import gradio as gr
 


load_dotenv() # Load environment variables from .env file. This is needed so that the right API keys are loaded. Otherwise, you are not able to call the model/servce. 
client = OpenAI()


"""
----------------------------------------------------Guardrails Class Section----------------------------------------------------
"""

"""
------Input Orchestrator Guardrails Class Section------
"""
class in_pii_and_poli_guardrail(BaseModel): #his class is for the input guardrail. Used to make sure the input does not contain any PII or Political content.
    political_and_pii: bool 
    "Checking if there are any political or personally identifiable information in the input about a person."
 



"""
------Output Internal Agent Guardrails Class Section------
"""

class product_eval_bias(BaseModel):
    product_eval_bias: bool 
    "Checking if there are any bias for the place where they are in the funding stages in the product evaluation."



class team_eval_bias(BaseModel):
    team_eval_bias: bool 
    "Checking if there are any bias on gender, age, or race, in the team evaluation."





"""
------Output Orchestrator Guardrails Class Section------
"""
class out_pii_and_poli_guardrail(BaseModel): 
    sens_cont: bool 
    "Measures if the content is sensitive or not"
 
class hallucination_guardrail(BaseModel): 
    hallucination: bool 
    "Measures if the content has hallucination or not"

class over_extreme_langauge_guardrail(BaseModel): 
    extreme_wording: bool 
    "Measures if the content has extreme wording or not"

class bias_guardrail(BaseModel): 
    bias: bool 
    "Measures if the content has bias wording or not"






"""
----------------------------------------------------Input guardrail section----------------------------------------------------
"""


"""
------Input Politiical and PII Guardrail------
"""
in_guardrail_agent = Agent(
    name="input_pii_poli_agent",
    instructions="When you receive a message, analyze it for any personally identifiable information (PII) or analyze it for any political content, which should be avoided. Either one needs to be true for the guardrail to be triggered.",
    output_type=in_pii_and_poli_guardrail,
    model="gpt-4o-mini"
)


"""
This function checks the input for any PII or Political content. If detected, the tripwire_triggered is triggered. 
"""
@input_guardrail 
async def pii_and_poli_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(in_guardrail_agent, input, context=ctx.context)
 
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.political_and_pii,
    )

 

"""
----------------------------------------------------Output Guardrail Section----------------------------------------------------
"""

"""
------Output Internal Agent Guardrails def Section------
"""

bias_in_product_eval_agent = Agent(
    name="bias_in_product_eval_agent",
    instructions="You are an expert agent responsible for detecting potential bias in the evaluation of a product, service, or business idea. "
        "Carefully examine the input for signs of biased assumptions, unfair comparisons, culturally insensitive language, or exclusionary framing. "
        "You should also flag any statements that show favoritism, lack objectivity, or reinforce stereotypes."
        "Your goal is to provide a reasoned analysis identifying whether bias is present, and if so, explain what kind of bias is detected "
        "and how it may affect the credibility or inclusivity of the evaluation. Be specific, concise, and constructive in your feedback.",
    output_type=product_eval_bias,
    model="gpt-4o-mini"
)


"""
This function checks the output for any bias in product evaluation. If detected, the tripwire_triggered is triggered. 
"""
@output_guardrail
async def bias_in_product_eval(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(bias_in_product_eval_agent, input, context=ctx.context)
 
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.product_eval_bias,
)


bias_in_team_eval_agent = Agent(
    name="bias_in_team_eval_agent",
    instructions="You are responsible for identifying potential bias in the evaluation of a startup team. "
        "Review the input for biased language, assumptions, or judgments related to gender, ethnicity, age, education background, geographic origin, or professional experience. "
        "Pay attention to whether the evaluation unfairly favors or criticizes individuals or groups based on identity factors or stereotypes."
        "Your goal is to determine whether bias is present in how the team is being described or assessed. "
        "If bias is detected, explain what type of bias it is, how it may influence the evaluation, and provide a constructive suggestion to improve objectivity and fairness.",
    output_type=team_eval_bias,
    model="gpt-4o-mini"
)


"""
This function checks the output for any bias in team evaluation. If detected, the tripwire_triggered is triggered. 
"""
@output_guardrail
async def bias_in_team_eval(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(bias_in_team_eval_agent, input, context=ctx.context)
 
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.team_eval_bias,
)


"""
------Orchestrator Guardrails def Section------
"""

#----Output Politiical and PII Guardrail----
out_pii_or_poli_guardrail_agent = Agent(
    name="sensitive_content_agent",
    instructions="When you receive a message, analyze it for any sensitive content. This " 
    "Includes but is not limited to: PII, political content, hate speech, or any " 
    "other content that should not be shared/can cause offense to a group of people.",
    output_type=out_pii_and_poli_guardrail,
    model="gpt-4o-mini"
)

"""
This function checks output for any sensitive content. If dectected, the tripwire_triggered is triggered. 
"""
@output_guardrail
async def sens_content_guardrail(
     ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(out_pii_or_poli_guardrail_agent, input, context=ctx.context)
 
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.sens_cont,
    )
 


#----Hallucinations Guardrail----
hallucination_guardrail_agent = Agent(
    name="hallucination_finder_agent",
    instructions="Check if the output includes hallucinations—claims or facts that are not grounded in the input, known context, or verified sources. Set hallucination = True only when clear inaccuracies or invented content are detected.",
    output_type=hallucination_guardrail,
    model="gpt-4o-mini"
)

"""
This function checks the output for any hallucinations within the output. If dectected, the tripwire_triggered is triggered. 
"""
@output_guardrail
async def  hallucination_output_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(hallucination_guardrail_agent, input, context=ctx.context)
 
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.hallucination,
    )
 


#----Extreme Language guardrail----
over_extreme_langauge_guardrail_agent = Agent(
    name="extreme_language_agent",
    instructions="Assess the output for overly extreme or emotionally charged language (e.g., hyperbole, sensationalism, inflammatory tone). Mark extreme_wording = True if the wording is disproportionate to the subject matter or could mislead, provoke, or escalate.",
    output_type=over_extreme_langauge_guardrail,
    model="gpt-4o-mini"
)

"""
This function checks the output for any extreme language within the output. If dectected, the tripwire_triggered is triggered. 
"""
@output_guardrail
async def  extreme_language_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(over_extreme_langauge_guardrail_agent, input, context=ctx.context)
 
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.extreme_wording,
    )


"""
----Bias guardrail----
"""
bias_guardrail_agent = Agent(
    name="bias_agent",
    instructions="Check the output for any form of bias—ideological, cultural, or demographic. If bias is detected, set bias = True and include representative examples in examples that highlight where and how the bias appears. Prioritize objectivity and fairness.",
    output_type=bias_guardrail,
    model="gpt-4o-mini"
)


"""
This function checks the output for any bias as a whole. If detected, the tripwire_triggered is triggered. 
"""
@output_guardrail 
async def  bias_detection_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(bias_guardrail_agent, input, context=ctx.context)
 
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.bias,
    )



"""
----------------------------------------------------Startup Evaluator Tool Agents Section----------------------------------------------------
"""
product_eval_agent = Agent(
    name="product_eval_agent",
    instructions="Evaluate the startup's product quality, tech defensibility, and fit to user needs. Score 0–5.",
    output_guardrails= [bias_in_product_eval]
)

market_eval_agent = Agent(
    name="market_eval_agent",
    instructions="Evaluate the market size, growth, timing, and competitive landscape. Score 0–5."
)

team_eval_agent = Agent(
    name="team_eval_agent",
    instructions="Evaluate the startup's founding and leadership team on experience, execution, and completeness. Score 0–5.",
    output_guardrails=[bias_in_team_eval]
)




"""
----------------------------------------------------Main Startup Evaluator Agent Section----------------------------------------------------
"""
startup_eval_agent = Agent(
    name="startup_eval_agent",
    instructions="Evaluate the startup's founding and leadership team on experience, execution, and completeness. Score 0–5.",
        tools=[
        product_eval_agent.as_tool(
            tool_name="product_evaluation_tool",
            tool_description="Evaluates product quality, defensibility, and user fit."
        ),
        market_eval_agent.as_tool(
            tool_name="market_evaluation_tool",
            tool_description="Analyzes market size, timing, and competition."
        ),
        team_eval_agent.as_tool(
            tool_name="team_evaluation_tool",
            tool_description="Assesses team experience, execution, and completeness."
        )
    ]
)

"""
-----------------Startup Improvements Handoff Agents Section-----------------
"""

differentiation_analyzer_agent = Agent( # Grabs the ideas of the given input. General concepts/aspects are taken which can be passed for analysis and summarization. 
    name="differentiation_analyzer_agent",
    instructions=(
        "Analyze the startup description to identify what differentiates it from existing competitors. "
        "Focus on unique features, technology, user experience, market positioning, or business model innovation. "
        "Clearly articulate the startup's competitive edge and highlight any elements that are replicable vs. defensible. "
        "If no clear differentiation is present, note that explicitly."
    )
)
 

vision_agent = Agent( # Conducts analysis of the text from the key ideas that were grabbed. Looks further into the details and importance of the ideas. 
    name="vision_agent",
    instructions=(
        "Examine the startup’s long-term vision and ambition. Assess how clearly the founder(s) communicate their goals, future market position, "
        "and broader impact. Determine whether the vision is compelling, realistic, and aligned with the current product or roadmap. "
        "Flag overly vague, generic, or inflated language, and comment on how the vision supports potential investor or user confidence."
    )
)
 

scalability_agent = Agent(# Develops a summary the text after grabbing key details/idea and general analysis. Provides a comprehensive output. 
    name="scalability_agent",
    instructions=(
        "Evaluate the startup’s potential for scale. Analyze whether its core product, business model, and operations can grow efficiently as demand increases. "
        "Consider factors like automation, infrastructure, network effects, marginal costs, team scalability, and platform readiness. "
        "Flag any structural barriers to scaling or signs that the current setup is not prepared for large-scale growth."
    )
)


"""
-----------------Startup Improvement Agent Section-----------------
"""
startup_improvement_agent = Agent(
    name="startup_improvement_agent",
    instructions=(
        "You are responsible for analyzing and improving a startup based on its differentiation, vision, and scalability. "
        "Use your specialized agents only if the input contains relevant details in that area. "
        "Handoff to:"
        "- `differentiation_analyzer_agent` if there are details about competition, features, or market positioning."
        "- `vision_agent` if the input mentions mission, ambition, or long-term goals."
        "- `scalability_agent` if the input discusses growth, infrastructure, or market expansion plans."
        "Return a combined response highlighting strengths, weaknesses, and opportunities for improvement."
    ),
    handoffs=[differentiation_analyzer_agent, vision_agent, scalability_agent]
)


"""
----------------------------------------------------Risk Finder Agent Section ----------------------------------------------------
"""
risk_finder_agent = Agent(
    name="risk_finder_agent",
    instructions="Scan the content for known startup risk factors. Flag any that are present, with a 1-line explanation. Examples could include: being a solo founder, No revenue model, High competition industry, Regulatory exposure, and High burn/low growth of cash."
)


"""
----------------------------------------------------Orchestrator Agent Section----------------------------------------------------
"""
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are responsible for coordinating the overall evaluation of a startup submission. "
        "You have access to a specialized agent called `startup_improvement_agent` which contains expert agents that handle different dimensions of startup analysis. "
        "Your task is to:"
        "- Review the user input for meaningful startup-related content."
        "- If the input seems to include information about differentiation, vision, or scalability, hand it off to `startup_improvement_agent`."
        "- `startup_improvement_agent` will internally decide which of its sub-agents to activate based on content relevance."
        "- If the input is vague, generic, or lacks business substance, you may return a brief summary or request clarification instead."
        "Return a clear, well-organized response summarizing the startup’s strengths, weaknesses, and areas for improvement, combining insights from any active agents."
    ),
    handoffs=[startup_improvement_agent],
    input_guardrails=[pii_and_poli_guardrail],
    output_guardrails=[sens_content_guardrail, hallucination_output_guardrail, extreme_language_guardrail, bias_detection_guardrail]
)
 



"""
----------------------------------------------------Async Main Section----------------------------------------------------
"""


async def evaluate_startup_gradio(user_input: str):
    try:
        result = await Runner.run(orchestrator_agent, input=user_input)
        risks = await Runner.run(risk_finder_agent, input=user_input)
        return result.final_output, risks.final_output or "No major risks flagged."
    except OutputGuardrailTripwireTriggered:
        return "Guardrail tripped – sensitive content detected. Please revise your input.", "" 

def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Startup Evaluation Assistant")
        gr.Markdown("Submit your startup idea below. Our platform will evaluate it and flag potential risks! Providing you with useful feedback on how to imporve it moving forward!")

        user_input = gr.Textbox(label="Startup Information", lines=10, placeholder="Describe your startup...")
        submit = gr.Button("Run Model")

        feedback_output = gr.Textbox(label="General Feedback", lines=10)
        risk_output = gr.Textbox(label="Critical Risk Factors", lines=5)

        submit.click(
            fn=evaluate_startup_gradio,
            inputs=user_input,
            outputs=[feedback_output, risk_output]
        )

    demo.queue()
    demo.launch()

if __name__ == "__main__":
        launch_ui()