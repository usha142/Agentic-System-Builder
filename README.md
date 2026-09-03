🤖 Agentic System Builder
Project Overview

Agentic System Builder is an AI-powered application developed using Python and Streamlit. It provides a single platform where users can interact with different specialized AI agents for software development tasks.

The system is designed to divide software development activities among different AI roles such as Coding, Testing, Debugging, Frontend Development, Backend Development, and Software Engineering.

Objectives
To build an interactive AI-based software development system.
To provide specialized AI agents for different development tasks.
To simplify coding, testing, and debugging activities.
To explore the concept of Agentic AI and multi-agent systems.
To provide an easy-to-use interface using Streamlit.
Features

The application provides the following AI teams and agents:

Coding Team

Helps users with programming and coding-related tasks.

Tester

Assists with software testing, validation, and identifying potential issues.

Debugger

Helps identify errors and provides possible solutions for debugging.

Software Engineer

Supports general software engineering and development activities.

Frontend Developer

Assists with frontend development tasks and user-interface-related requirements.

Backend Developer

Assists with backend development and server-side programming tasks.

Technologies Used
Python
Streamlit
OpenRouter
Artificial Intelligence
Large Language Models (LLMs)
JSON
CSV
SQLite
Git
GitHub
How the System Works

The user first opens the Streamlit application and selects the required AI agent or development team.

The user then provides a request or task. The selected agent processes the request using the configured OpenRouter model and generates an appropriate response.

The general workflow is:

User Request → Select Agent → AI Processing → Response

Project Structure

The project contains Python source files, configuration files, CSV data files, a JSON configuration file, and a SQLite database.

Main files include:

aai.py – Main application file.
agent_systems.json – Agent system configuration.
drl.py – Python module used in the project.
instructions.txt – Project instructions.
workspace/script.py – Workspace Python script.
automation.db – Local SQLite database.
CSV files – Team and project-related data.
.gitignore – Specifies files that should not be tracked by Git.
Installation

Clone the repository from GitHub:

git clone https://github.com/usha142/agentic-system-builder.git

Open the project directory:

cd agentic-system-builder

Create a virtual environment:

python -m venv agents

Activate the virtual environment on Windows:

agents\Scripts\activate

Install the required dependencies.

Running the Application

Run the Streamlit application using:

streamlit run aai.py

After running the command, open the Streamlit application in a web browser.

The application can normally be accessed at:

http://localhost:8501

OpenRouter API

The application uses OpenRouter to access AI models.

Users can provide their OpenRouter API key through the application interface.

Security: API keys and other confidential information should never be uploaded to a public GitHub repository.

Applications and Use Cases

This project can be useful for:

AI-assisted programming
Software development
Code generation
Software testing
Debugging
Frontend development
Backend development
Learning Agentic AI
Understanding multi-agent systems
Experimenting with AI-powered development workflows
Future Enhancements

The project can be further improved by adding:

More specialized AI agents.
Agent-to-agent communication.
Conversation history.
Persistent AI memory.
Automated testing.
Code execution and validation.
GitHub integration.
Project file management.
Agent performance monitoring.
Advanced dashboard features.
Cloud deployment.
Project Status

Active Development

This project is developed as an exploration of Agentic AI and AI-assisted software development.

Author

Lakshmi Usha Sree

GitHub: usha142

Conclusion

Agentic System Builder demonstrates how AI agents can be organized into specialized roles to support different software development activities. By combining Python, Streamlit, OpenRouter, and Agentic AI concepts, the project provides an interactive environment for exploring AI-powered software engineering.