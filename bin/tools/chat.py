from sophia.llm_client import LLMClient

def main():
    print("Initializing Sophia (Checkpoint 150)...")
    client = LLMClient()
    
    print("\n--- Sophia AI Interview Coach (Type 'quit' to exit) ---")
    print("Test her knowledge on: RAG, Deployment, Loss Functions, or ML Basics.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            response = client.ask(question=user_input, topic="General", notes=[])
            
            print(f"\nSophia: {response.technical_answer}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

