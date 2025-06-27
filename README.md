# 💰 FinFluent - Your AI-Powered Financial Assistant

FinFluent is an intelligent personal finance management application that combines AI-powered financial advice with comprehensive expense tracking. Built with Streamlit and powered by OpenAI, it helps you manage your finances through natural language interactions and intuitive visualizations.

## ✨ Features

- **AI Financial Assistant**: Get personalized financial advice using natural language
- **Expense Tracking**: Log and categorize your income and expenses
- **Interactive Dashboard**: Visualize your spending patterns and financial health
- **Profile Management**: Set and track financial goals and personal details
- **RAG-Powered Insights**: Get context-aware financial advice using Retrieval-Augmented Generation
- **Responsive Design**: Works seamlessly on both desktop and mobile devices

## 🎯 Key Technologies

- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT, LangChain
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Streamlit Charts
- **Vector Database**: FAISS (for RAG implementation)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- (Optional) Perplexity API key for enhanced web search

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/finflo.git
   cd finflo
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PERPLEXITY_API_KEY=your_perplexity_api_key_here  # Optional
   ```
   
   Get your API keys:
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [Perplexity API Key](https://www.perplexity.ai/) (optional)

## 🖥️ Running the Application

### Web Interface (Recommended)
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`

### Key Features in Action

1. **Dashboard**
   - View your financial overview at a glance
   - Track income, expenses, and savings
   - Visualize spending patterns with interactive charts

2. **AI Assistant**
   - Ask financial questions in natural language
   - Get personalized advice based on your financial profile
   - Example queries:
     - "How can I save more money?"
     - "What's my current financial health?"
     - "Help me create a budget"
     - "Suggest investment options for my risk profile"

3. **Profile Management**
   - Update your personal information
   - Set and track financial goals
   - Configure notification preferences

## 🔧 Troubleshooting

- If you encounter API key errors, ensure your `.env` file is properly configured
- For RAG-related issues, check that the required PDF documents are in the `knowledge_base` directory
- Make sure all dependencies are installed using `pip install -r requirements.txt`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit and OpenAI
- Icons by [Font Awesome](https://fontawesome.com/)
- Color scheme inspired by modern fintech applications
