# app.py
from flask import Flask, request, jsonify
from langchain_utils import get_llm_chain

app = Flask(__name__)
# Initialize the chain once (you can also lazy-load it if needed)
chain = get_llm_chain()


@app.route("/generate-schema", methods=["POST"])
def generate_schema():
    """
    Expects a JSON body: { "description": "some requirement" }
    Returns a JSON: { "schema": "...model output..." }
    """
    try:
        data = request.get_json()
        user_requirement = data.get("description", "")
        # Generate the schema from user requirement
        result = chain.run(user_requirement=user_requirement)

        # The result is presumably a JSON string.
        # Optionally, we can attempt to parse it to confirm validity, then re-dump as JSON.
        # For simplicity, we'll just return it as a string.
        return jsonify({"schema": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run Flask in debug mode or production mode as you prefer
    app.run(host="0.0.0.0", port=5000, debug=True)
