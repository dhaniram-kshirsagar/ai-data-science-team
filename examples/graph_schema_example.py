"""
Example usage of the Graph Schema Agent for generating Neo4j graph schemas from CSV data.
"""
import os
import json
from langchain_openai import ChatOpenAI
from ai_data_science_team.agents.graph_schema_agent import GraphSchemaAgent

def main():
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return

    # Get absolute path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(current_dir), "data", "churn_data.csv")  # Changed to employees dataset

    print(f"\nProcessing CSV file: {csv_path}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        api_key=api_key,
        temperature=0  # Set to 0 for more consistent schema generation
    )
    
    # Create the agent
    agent = GraphSchemaAgent(
        model=llm,
        csv_path=csv_path,
        log=True,
        log_path="logs"
    )
    
    print("\nStarting Graph Schema Agent workflow...")
    
    # Run the agent
    response = agent.invoke_agent()
    
    # Show validation results
    validation = agent.get_validation_results()
    if validation:
        if validation.get("warnings"):
            print("\nData Quality Warnings:")
            for warning in validation["warnings"]:
                print(f"- {warning}")
    
    # Get results
    schema = agent.get_schema()
    if schema:
        print("\nInferred Schema:")
        print(json.dumps(schema, indent=2))
        
        cypher = agent.get_cypher()
        if cypher:
            print("\nGenerated Cypher Statements:")
            print(cypher)
            
        print("\nSchema Generation Complete!")
    else:
        print("\nError: Failed to generate schema. Check the logs for details.")

if __name__ == "__main__":
    main()
