GAIA Benchmarking Agent

A Python-based benchmarking agent for the Hugging Face Agent Course, designed to automate GAIA task evaluation and answer submission.

Project Overview

This project provides a simple framework for working with GAIA benchmark tasks through a Gradio interface.

The agent supports:

Fetching GAIA task attachments and identifying their file types, such as Excel files.
Processing and handling task-related files.
Handling errors and displaying stack traces for debugging.
Running GAIA tasks and generating answers.
Submitting answers for evaluation.
Displaying task questions, generated answers, and evaluation status through a Gradio interface.
Requirements
Python 3.11.9
Hugging Face account
Installation
1. Clone the repository
git clone https://github.com/your-username/gaia-benchmark-agent.git
cd gaia-benchmark-agent
2. Create a virtual environment
python -m venv venv

Activate the virtual environment:

Linux/macOS

source venv/bin/activate

Windows

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the application
python app.py

The Gradio interface will start locally. Open the URL displayed in the terminal.

Usage
Start the application using python app.py.
Login to your Hugging Face account through the Gradio interface.
Click Run Evaluation & Submit All Answers.
The agent will process the available GAIA tasks.
Generated answers will be submitted for evaluation.
Review the run status, questions, and generated answers in the interface.
Gradio Interface

The Gradio interface provides a simple way to interact with the benchmarking agent.

It includes:

Login: Authenticate with your Hugging Face account.
Run Evaluation & Submit All Answers: Start the GAIA evaluation process.
Run Status: Display the current execution status and submission results.
Questions & Answers: Display GAIA questions alongside the generated answers.
Testing

You can test the agent locally by running:

python app.py

For development and debugging, you can:

Submit a single GAIA task.
Process multiple tasks.
Check generated answers.
Review error messages and stack traces.
Verify the submission status.
Project Structure
gaia-benchmark-agent/
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
└── ...
Contributing

Contributions are welcome.

To contribute:

Fork the repository.
Create a new branch.
git checkout -b feature/your-feature
Make your changes.
Test the changes locally.
Commit your changes.
git commit -m "Add your feature"
Push the branch and open a Pull Request.

Please make sure your changes are tested and clearly documented.

License

This project is licensed under the MIT License.

See the LICENSE file for the complete license text.