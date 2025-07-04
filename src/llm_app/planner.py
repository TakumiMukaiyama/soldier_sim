from typing import Any, Dict, List

from langchain.prompts import PromptTemplate

from .schemas import PlanOutput


class PydanticChain:
    """Simple implementation of PydanticChain for structured output"""

    def __init__(self, llm, output_model, prompt):
        """
        Initialize the PydanticChain

        Args:
            llm: LLM client to use
            output_model: Pydantic model to validate output
            prompt: Prompt template or string to use
        """
        self.llm = llm
        self.output_model = output_model

        if isinstance(prompt, str):
            # Convert string to PromptTemplate
            self.prompt = PromptTemplate(
                template=prompt,
                input_variables=["agent_state", "reflective_memory", "poi_list"],
            )
        else:
            self.prompt = prompt

    def invoke(self, input_values: Dict[str, Any]) -> Any:
        """
        Invoke the chain with input values

        Args:
            input_values: Dictionary of input values for the prompt

        Returns:
            Instance of the output model
        """
        # Format prompt with input values
        prompt_value = self.prompt.format(**input_values)

        # Get JSON example for output model
        json_schema = self.output_model.model_json_schema()
        example = self._generate_example_from_schema(json_schema)

        # Invoke LLM with structured output instructions
        system_prompt = "You are an AI planner for a military simulation. Generate a plan for the agent based on their state, memory, and available POIs."
        result = self.llm.invoke_with_structured_output(
            system_prompt=system_prompt,
            user_prompt=prompt_value,
            json_structure=example,
        )

        # Parse and validate with Pydantic model
        return self.output_model(**result)

    def _generate_example_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an example dictionary from a JSON schema"""
        example = {}
        properties = schema.get("properties", {})

        for name, prop in properties.items():
            if prop.get("type") == "string":
                if name == "agent_id":
                    example[name] = "agent_1"
                elif name == "chosen_poi":
                    example[name] = "training_ground_1"
                elif name == "activity":
                    example[name] = "train"
                elif name == "reason":
                    example[name] = "Energy levels optimal for training"
                else:
                    example[name] = "example_string"
            elif prop.get("type") == "integer":
                example[name] = 2
            elif prop.get("type") == "number":
                example[name] = 0.5
            elif prop.get("type") == "boolean":
                example[name] = True
            elif prop.get("type") == "array":
                example[name] = []
            elif prop.get("type") == "object":
                example[name] = {}

        return example


class Planner:
    """Planner for agent decisions using PydanticChain and LLMs"""

    def __init__(self, llm_client):
        """
        Initialize the Planner

        Args:
            llm_client: LLM client to use (Gemini or Azure)
        """
        self.chain = PydanticChain(
            llm=llm_client, output_model=PlanOutput, prompt=self._build_prompt()
        )

    def _build_prompt(self) -> str:
        """Build the prompt template for the planner"""
        return """
You are a planner for a military simulation agent. Your task is to decide the agent's next action based on their current state, memory, and available Points of Interest (POIs).

Agent State:
{agent_state}

Reflective Memory (Agent's past experiences and patterns):
{reflective_memory}

Available POIs:
{poi_list}

Based on this information, determine:
1. Which POI the agent should visit next
2. What activity they should perform there
3. How long they should spend (in hours, 1-8)
4. Why this is the optimal choice given their current state and needs

Return your decision in this exact JSON format:
{
  "agent_id": "the agent's ID",
  "chosen_poi": "ID of the chosen POI",
  "activity": "train/eat/rest/manage/arm",
  "expected_duration": 2,
  "reason": "Brief explanation of why this is the optimal choice"
}
"""

    def plan_action(
        self,
        agent_state: Dict[str, Any],
        reflective_memory: Dict[str, Any],
        poi_list: List[Dict[str, Any]],
    ) -> PlanOutput:
        """
        Plan the next action for an agent

        Args:
            agent_state: Current state of the agent
            reflective_memory: Agent's reflective memory
            poi_list: List of available POIs

        Returns:
            PlanOutput object with the planned action
        """
        # Convert to string representations for the prompt
        prompt_vars = {
            "agent_state": self._format_dict(agent_state),
            "reflective_memory": self._format_dict(reflective_memory),
            "poi_list": self._format_list(poi_list),
        }

        # Invoke the chain
        return self.chain.invoke(prompt_vars)

    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format a dictionary as a readable string"""
        if not d:
            return "No data available"

        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for k2, v2 in v.items():
                    lines.append(f"  - {k2}: {v2}")
            elif isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{k}: {v}")

        return "\n".join(lines)

    def _format_list(self, lst: List[Dict[str, Any]]) -> str:
        """Format a list of dictionaries as a readable string"""
        if not lst:
            return "No items available"

        result = []
        for i, item in enumerate(lst):
            result.append(f"POI {i + 1}:")
            for k, v in item.items():
                if isinstance(v, dict):
                    result.append(f"  {k}:")
                    for k2, v2 in v.items():
                        result.append(f"    - {k2}: {v2}")
                elif isinstance(v, list):
                    result.append(f"  {k}: {v}")
                else:
                    result.append(f"  {k}: {v}")
            result.append("")

        return "\n".join(result)
