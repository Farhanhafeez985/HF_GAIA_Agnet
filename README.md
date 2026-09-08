**README.md**

**GAIA Benchmarking Agent**

A Python-based benchmarking agent for Hugging Face agents, designed for the Hugging Face Agent Course.

**Project Overview**

This project provides a framework for interacting with the Hugging Face Agent Course, including:

* Fetching task attachments and determining their type (e.g., Excel files)
* Handling errors and printing stack traces for debugging purposes
* Submitting answers to GAIA tasks and evaluating the results

**Requirements**

* Python 3.11.9

**Installation**

1. Clone this repository using `git clone https://github.com/your-username/gaia-benchmark-agent.git`
2. Install the required libraries using `pip install -r requirements.txt`
3. Run the application using `python app.py`

**Usage**

1. Login to your Hugging Face account
2. Click **Run Evaluation & Submit All Answers**
3. The agent will process all GAIA tasks and submit the answers
4. View the run status and submission result using the Gradio interface

**Gradio Interface**

The Gradio interface provides a simple and intuitive way to interact with the agent. It includes:

* A login button to authenticate with your Hugging Face account
* A run button to submit all answers
* A text box to display the run status and submission result
* A table to display the questions and agent answers

**Testing**

You can test the agent locally by running the `app.py` file and submitting a single task. You can also test the agent's functionality by submitting multiple tasks and viewing the results.

**Contributing**

Contributions to this project are welcome. Please submit pull requests with any changes or improvements you'd like to make.

**License**

This project is licensed under the MIT License. See the `LICENSE` file for details.