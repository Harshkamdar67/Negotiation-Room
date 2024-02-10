
from flask import Flask, render_template, request, session, redirect, url_for, copy_current_request_context
from flask_socketio import join_room, leave_room, send, SocketIO
import random
import torch
import markdown
from transformers import BertTokenizer, BertForSequenceClassification
import google.generativeai as genai
import string
from threading import Thread

print('app starting')

app = Flask(__name__)
app.config["SECRET_KEY"] = "hjhjsdahhds"
socketio = SocketIO(app, cors_allowed_origins="*", engineio_logger=True, logger=True, async_mode='eventlet')


rooms = {}

def generate_unique_code(length):
    while True:
        characters = string.ascii_letters + string.digits
        code = "".join(random.choice(characters) for _ in range(length))
        
        if code not in rooms:  
            break
    
    return code

@app.route("/", methods=["POST", "GET"])
def home():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")
        code = request.form.get("code")
        create = request.form.get("create", False)

        if not name:
            return {"error": "Please enter a name."}, 400

        if create and not code:
            room, new_var = new_func()
            new_var
            rooms[room] = {"members": 0, "messages": []}
            session["room"] = room
            session["name"] = name
            return {"redirect": url_for("room")}

        elif code in rooms:
            session["room"] = code
            session["name"] = name
            return {"redirect": url_for("room")}
        else:
            return {"error": "Room does not exist."}, 404

    return render_template("home.html")

def new_func():
    new_var = room = generate_unique_code(10)
    return room,new_var


@app.route("/room")
def room():
    room = session.get("room")
    if room is None or session.get("name") is None or room not in rooms:
        return redirect(url_for("home"))

    return render_template("room.html", code=room, messages=rooms[room]["messages"])


certainty_model_path = 'models/certainty_prediction_model'
certainty_model = BertForSequenceClassification.from_pretrained(certainty_model_path)
certainty_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
certainty_model.to(device)

# phi2_model_name = "models/phi-2"
# phi2_model = AutoModelForCausalLM.from_pretrained(phi2_model_name)
# phi2_tokenizer = AutoTokenizer.from_pretrained(phi2_model_name, trust_remote_code =True)
# phi2_pipeline = pipeline("text-generation", model=phi2_model, tokenizer=phi2_tokenizer)
# phi2_model.to(device)

# SYSTEM_PROMPT = """
# Analyze a negotiation dialogue to identify non-negotiable terms and suggest alternative strategies.
# """

# generation_config = GenerationConfig.from_pretrained(phi2_model_name)
# generation_config.max_new_tokens = 512
# generation_config.temperature = 0.001
# generation_config.do_sample = True

# streamer = TextStreamer(phi2_tokenizer, skip_prompt=True, skip_special_tokens=True)

# llm = pipeline(
#     "text-generation",
#     model=phi2_model,
#     tokenizer=phi2_tokenizer,
#     # max_new_tokens = 512,
#     eos_token_id=phi2_tokenizer.eos_token_id,
#     pad_token_id=phi2_tokenizer.eos_token_id,
#     streamer=streamer,
#     )

# def create_suggestion_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT)-> str:
#     """
#     Creates a prompt for the phi-2 model to generate alternative negotiation strategies based on the conversation.

#     Parameters:
#     conversation (list of str): The conversation history, where each item is a message string.

#     Returns:
#     str: A prompt string for the phi-2 model.
#     """
#     if not system_prompt:
#         return cleandoc(
#             f"""
#         Instruct: {prompt}
#         Output:
#         """
#         )
#     return cleandoc(
#         f"""
#         Instruct: {system_prompt} {prompt}
#         Output:
#         """
#     )


CERTAINTY_SEQUENCE_THRESHOLD = 1
last_active_user = {}


def background_task(func, *args, **kwargs):
    """ Run a background task in a separate thread. """
    thread = Thread(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread



@socketio.on("message")
def handle_message(data):
    print("calling handle_message function")
    room = session.get("room")
    if room not in rooms:
        return
    

    user = session.get("name")
    message_text = data["data"]
    
    last_active_user[room] = user
    
    if "messages_by_user" not in rooms[room]:
        rooms[room]["messages_by_user"] = {}
    if "certainty_history_by_user" not in rooms[room]:
        rooms[room]["certainty_history_by_user"] = {}

    if user not in rooms[room]["messages_by_user"]:
        rooms[room]["messages_by_user"][user] = []
    if user not in rooms[room]["certainty_history_by_user"]:
        rooms[room]["certainty_history_by_user"][user] = []

    rooms[room]["messages_by_user"][user].append(message_text)
    
    all_messages = sum(rooms[room]["messages_by_user"].values(), [])
    print('all messages :' , all_messages)
    
    # @copy_current_request_context
    # def background_generate_middle_ground(all_messages, room):
    #     generate_middle_ground(all_messages, room)

    # # Create a copy of the current request context to use in the background task
    # @copy_current_request_context
    # def background_generate_and_send_suggestion(user_messages, room, user):
    #     generate_and_send_suggestion(user_messages, room, user)
        
    # background_task(background_generate_middle_ground, all_messages, room)
    
    generate_middle_ground(all_messages, room)
    
    user_messages = rooms[room]["messages_by_user"][user]
    overall_certainty, individual_predictions = predict_certainty_conversation(user_messages, certainty_model, certainty_tokenizer)
    rooms[room]["certainty_history_by_user"][user].append(overall_certainty)
    print('overall certainity ', overall_certainty)
    for user, certainty_history in rooms[room]["certainty_history_by_user"].items():
        # if len(certainty_history) >= CERTAINTY_SEQUENCE_THRESHOLD and all(certainty_history[-CERTAINTY_SEQUENCE_THRESHOLD:]):
            # if len(certainty_history) == CERTAINTY_SEQUENCE_THRESHOLD or not all(certainty_history[-CERTAINTY_SEQUENCE_THRESHOLD-1:-1]):
        if len(certainty_history) >= CERTAINTY_SEQUENCE_THRESHOLD:
            generate_and_send_suggestion(user_messages, room, user)

    send({"name": session.get("name"), "message": message_text, "certainty": "Certain" if overall_certainty else "Not Certain"}, to=room)

    
def generate_and_send_suggestion(user_messages, room, user):
    print('calling function generate prompt')
    conversation_context = " ".join(user_messages[:-1]) if len(user_messages) > 1 else "The conversation has focused primarily on price negotiations."
    non_negotiable_statement = user_messages[-1]
    
    genai.configure(api_key="YOUR-API-KEY")
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Given the negotiation context below, identify the key negotiable term from the most recent statement and suggest an alternative strategy other than the negotiable term. The strategy should consider other aspects or terms that could be negotiable, aiming for a win-win outcome. It's important to communicate the suggestions in a way that indicates possibilities rather than certainties. Keep the responses short.

    Previous negotiation context:
    {conversation_context}

    Most recent, non-negotiable statement:
    {non_negotiable_statement}

    Task:
    firstly analyse the statements and decide whether an intervention is necessary at this point or not, if yes then :
        1. Identify the term which can be negotiated upon from the most recent statement.
        2. If no negotiable term is there, just respond with "No comment". Otherwise, Propose an alternative negotiation strategy other than the negotiable term while addressing the negotiation's key aspects.
        3. keep the responses short such that they are easy and fast to read for the user.
    """

    
    config = {
        "max_output_tokens": 216,
        "temperature" : 0.3
    }
    responses = model.generate_content(prompt, generation_config=config)
    
    text_responses = []

    # Iterate through each candidate in the responses
    for candidate in responses.candidates:
        # Concatenate the text of all parts for this candidate
        candidate_text = ''.join(part.text for part in candidate.content.parts)
        
        # Format the text using HTML for better structure
        formatted_response = format_response_as_html(candidate_text)
        text_responses.append(formatted_response)
    
    final_response_html = '<hr>'.join(text_responses)
    
    print('Suggestions generated by model (HTML formatted):', final_response_html)
    if "No comment" not in responses.text:
        formatted_response = to_html(responses.text)
        # Find the target user as the user who is not the last active user
        target_user = next((user for user in user_sessions[room] if user != last_active_user.get(room)), None)
        
        if target_user:
            target_session_id = user_sessions[room].get(target_user)
            if target_session_id:
                # Send the suggestion to the target user
                send({"name": "System", "message": formatted_response, "messageType": "suggestion"}, room=target_session_id)
                print(f"Suggestion sent to {target_user}.")
            else:
                print(f"Session ID for {target_user} not found.")
        else:
            print("No suitable target user found in the room.")

def to_html(markdown_text):
    html_content = markdown.markdown(markdown_text)
    left_aligned_html = f'{html_content}'
    return left_aligned_html



def generate_middle_ground(user_messages, room):
    print('middle ground function has been called')
    genai.configure(api_key="YOUR-API-KEY")
    model = genai.GenerativeModel('gemini-pro')
    
    # Combine user messages into a single string for analysis
    combined_terms = '\n\n'.join(user_messages)
    
    prompt = f"""
        Given a negotiation between a seller and a buyer, we aim to mediate and find a mutually beneficial compromise based solely on the terms provided by each party. Before intervening, it's crucial to determine if enough information has been shared to propose a meaningful compromise.

        Seller and Buyer have provided the following terms:

        {combined_terms}

        Based on the information provided:
        1. If there are sufficient terms from both parties, analyze the terms presented by both the seller and the buyer and find the key differences and potential areas for compromise. If not enough terms, just respond with "No comment".
        2. Respond with a new set of terms that strictly incorporate elements from both parties' preferences, considering constraints such as price and additional features. 
        3. The output should  be proposed compromise which should be concise, unbiased and strictly derived from the provided terms of both parties, without introducing suggestions outside their stated terms.

        Proposed Compromise or Silence if Insufficient Information:
        
        output:
        give only the proprosed compromise as the output the user doesnt need to see all the decisions you make 
        """


    # Assuming genai and model are configured as shown earlier
    config = {
        "max_output_tokens": 216,
        "temperature": 0.3,
    }
    response = model.generate_content(prompt)

    if "No comment" not in response.text:
        formatted_response = to_html(response.text)
        send({"name": "System", "message": formatted_response, "messageType": "middle-ground"}, to=room, html=True)
    else:
        print('not enough information')

def format_response_as_html(text):
    """
    Converts a structured text response into HTML with left-aligned text for better readability.
    """
    html_response = '<div style="text-align: left;">'
    for line in text.split('\n'):
        if line.strip():  
            html_response += f'<p>{line}</p>'
    html_response += '</div>'
    return html_response
    
def predict_certainty_conversation(conversation, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cumulative_predictions = []
    weights = [0.5 ** i for i in range(len(conversation))]
    weights.reverse()

    for dialogue in conversation:
        inputs = tokenizer(dialogue, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1)
        cumulative_predictions.append(predicted_class.item())

    if len(cumulative_predictions) > 1:
        weighted_average = sum([a * b for a, b in zip(cumulative_predictions[:-1], weights[:-1])]) / sum(weights[:-1])
    else:
        weighted_average = 0  

    if cumulative_predictions[-1] == 1:  
        overall_certainty = 1
    else:
        overall_certainty = 1 if weighted_average > 0.5 else 0

    return overall_certainty, cumulative_predictions

    
user_sessions = {}  

@socketio.on('connect')
def on_connect():
    session_id = request.sid  
    room = session.get("room")
    name = session.get("name")
    if room and name:
        join_room(room)
        if room not in user_sessions:
            user_sessions[room] = {}
        user_sessions[room][name] = session_id
        print('users in a session ' , user_sessions)
        send({"name": name, "message": "has entered the room"}, room=room)

@socketio.on("disconnect")
def disconnect():
    room = session.get("room")
    name = session.get("name")
    leave_room(room)
    if room in rooms:
        rooms[room]["members"] -= 1
        if rooms[room]["members"] <= 0:
            del rooms[room]
    
    send({"name": name, "message": "has left the room"}, to=room)
    print(f"{name} has left the room {room}")

if __name__ == "__main__":
    socketio.run(app,port=8080)
