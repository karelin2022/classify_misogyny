from flask import Flask, request, session, render_template, redirect, url_for
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
 
import openai
from pathlib import Path
import time


# Uncomment the code below and add organization ID.
# openai.organization = ""

# Uncomment the code below and add OpenAI API key.
# openai.api_key = ""


UPLOAD_FOLDER = os.path.join('static', 'uploads')
 
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure upload folder path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Uncomment the code below and create secret key to enable session (can be anything).
# app.secret_key = ""


@app.route("/misogyny_calculator", methods=("GET", "POST"))
def misogyny_calc():
    """
    Classify text as having hostile, ambivalent, benevolent, or no misogyny.
    """

    if request.method == "POST":
        text = request.form["text_to_score"]

        # Remove newlines from text.
        if "\n" in text:
            text = text.strip()

        try_again = "Please try again."

        # If text is empty (e.g., just spaces), tell user to try again.
        if (text.isspace()):
            return redirect(url_for("misogyny_calc", missing_info=try_again))
        
        # Otherwise, give user the misogyny classification for the text.
        else:
            prompt = misogyny_classification(text)

            response = openai.Completion.create(model="text-davinci-003",
                                                prompt=prompt,
                                                temperature=1,
                                                max_tokens=2000,
                                                echo=True)

            misogyny_phrase = response["choices"][0]["text"].split("\n\n")[1]

            return redirect(url_for("misogyny_calc",
                                    typed_text = text, 
                                    output = misogyny_phrase))
        
    try_again = request.args.get("missing_info")
    original_text = request.args.get("typed_text")
    misogyny_label = request.args.get("output")

    if try_again:
        return render_template("misogyny_calculator.html", missing_info=try_again)
    elif misogyny_label:
        df = pd.DataFrame({'text': [original_text],
                           'classification': [misogyny_label]})

        csv_file = "app.csv"
        path = Path(csv_file)

        # This ensures all texts and classifications are saved to the csv file to check accuracy.
        if path.is_file():
            df.to_csv(csv_file, mode='a', index=False, header=False)
        else:
            df.to_csv(csv_file, index=False)

        return render_template("misogyny_calculator.html",
                               typed_text = original_text, 
                               output = misogyny_label)
    
    # This is what's first seen when the URL loads up.
    else:
        return render_template("misogyny_calculator.html")


def misogyny_classification(text):
    return """Hostile misogyny occurs when people hold views that are hostile and sexist against women and view women as manipulative, deceitful, capable of using seduction to control men, or needing to be kept in their place. Benevolent misogyny occurs when people frame women as innocent, pure, caring, nurturing, beautiful, or fragile and in need of protection. Ambivalent misogyny is a combination of benevolent and hostile misogyny, and ambivalent misogyny occurs when people view women as good, pure, and innocent, as well as manipulative or deceitful. No misogyny means there is no reference to women, or it’s a positive view of women. Is the text: {} classified as No Misogyny, Benevolent Misogyny, Ambivalent Misogyny, or Hostile Misogyny? Please output either "No Misogyny", "Benevolent Misogyny", "Ambivalent Misogyny", or "Hostile Misogyny". Please output the two words first with a period. Then give a short explanation of why.""".format(text)



@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    """
    Output misogynistic texts from a CSV file with misogyny classifications.
    """

    uploaded_df_html = ""
    if request.method == 'POST':
        if request.form["btn"]=="Upload":
            
            # Get csv file
            uploaded_df = request.files['uploaded_csv_file']

            # Get uploaded csv file name
            data_filename = secure_filename(uploaded_df.filename)

            # Save file to uploads folder in static folder
            uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))

            # Store uploaded csv file path in a flask session
            session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

            # Get uploaded file path from session
            data_file_path = session.get('uploaded_data_file_path', None)
        
            uploaded_df = pd.read_csv(data_file_path)

            if 'text' in uploaded_df.columns:

                # Drop rows without text.
                uploaded_df = uploaded_df.dropna(subset=['text'])

                y_pred = []

                for text in uploaded_df['text']:
                    if len(y_pred) in list(range(20, len(uploaded_df), 20)):
                        time.sleep(60)

                    prompt = misogyny_texts_csv(text)


                    response = openai.Completion.create(model="text-davinci-003",
                                                        prompt=prompt,
                                                        temperature=1,
                                                        max_tokens=2000,
                                                        echo=True)

                    misogyny_class = response["choices"][0]["text"].split("\n")[-1]

                    y_pred.append(misogyny_class)

                uploaded_df['misogyny_score'] = y_pred

                uploaded_df = uploaded_df[['text', 'misogyny_score']]

                uploaded_df['text'] = uploaded_df['text'].replace('\\n', ' ', regex=True)

                uploaded_df = uploaded_df[uploaded_df['misogyny_score'] != "No Misogyny."]

                uploaded_df = uploaded_df[uploaded_df['misogyny_score'] != "NoMisogyny"]

                uploaded_df = uploaded_df[uploaded_df['misogyny_score'] != "No Misogyny"].sort_values('misogyny_score', ascending=True)            

                if uploaded_df.empty:
                    uploaded_df_html = "No texts are misogynistic."
                else:
                    # DataFrame of misogynistic texts and classifications are transformed into a HTML table.
                    uploaded_df_html = uploaded_df.to_html(index=False)

            else:
                uploaded_df_html = 'CSV file needs column text.'

    return render_template('misogyny_texts.html', csv_data = uploaded_df_html)


def misogyny_texts_csv(text):
    return """Hostile misogyny occurs when people hold views that are hostile and sexist against women and view women as manipulative, deceitful, capable of using seduction to control men, or needing to be kept in their place. Benevolent misogyny occurs when people frame women as innocent, pure, caring, nurturing, beautiful, or fragile and in need of protection. Ambivalent misogyny is a combination of benevolent and hostile misogyny, and ambivalent misogyny occurs when people view women as good, pure, and innocent, as well as manipulative or deceitful. No misogyny means there is no reference to women, or it’s a positive view of women. Is the text: {} classified as No Misogyny, Benevolent Misogyny, Ambivalent Misogyny, or Hostile Misogyny? Please output either "No Misogyny", "Benevolent Misogyny", "Ambivalent Misogyny", or "Hostile Misogyny". Please output only two words.""".format(text)



if __name__ == '__main__':
    app.run(port=80, host='0.0.0.0', debug=True)