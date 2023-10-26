import { pinecone } from './pinecone-client.js'
import express from "express";
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from './config/pinecone.js'
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { makeChain } from './makechain.js'
import OpenAI  from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const app = express()
const port = 3000

const storage = {}

app.get('/answer', async (req, res) => {
  const question = req.query.q
  const history = req.query.history

  console.log(question)
  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }

  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  try {
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    /* create vectorstore*/
    const vectorStore = await PineconeStore.fromExistingIndex(
      new OpenAIEmbeddings({}),
      {
        pineconeIndex: index,
        textKey: 'text',
        namespace: PINECONE_NAME_SPACE, //namespace comes from your config folder
      },
    );

    //create chain
    const chain = makeChain(vectorStore);
    //Ask a question using chat history
    const response = await chain.call({
      question: sanitizedQuestion,
      chat_history: history || []
    });

    console.log('response', response);
    res.status(200).json(response);
  } catch (error) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
})

app.get('/user/:userId/tasks', async (req, res) => {
  const userId = req.params.userId
  const query = req.query.q
  const messages = []
  if (storage[userId] === undefined) {
    messages.push({ role: 'user', content: `I want you to maintain a todo list for me. Whatever task I need to perform, you need to add them in the todo list and once I complete them, you need to mark them as completed and delete them from the todo list.Currently I don't have any task in my todo list.` })
  } else {
    messages.push({ role: 'user', content: `I want you to maintain a todo list for me. Whatever task I need to perform, you need to add them in the todo list and once I complete them, you need to mark them as completed and delete them from the todo list.${storage[userId].replaceAll('Your', 'My')}` })
    // messages.push(...storage[userId])
  }

  messages.push({ role: 'user', content: `Question: ${query}` })

  console.log(JSON.stringify(messages))

  const chatCompletion = await openai.chat.completions.create({
    messages: messages,
    model: 'gpt-4',
    n: 1,
    user: userId,
    temperature: 0.2
  });

  console.log(chatCompletion.choices);

  const currentTaskMessages = [...messages, chatCompletion.choices[0].message, {"role":"user","content":"Question: what is the current state of my todo list?"}]
  console.log(currentTaskMessages)
  const currentTasksResponse = await openai.chat.completions.create({
    messages: currentTaskMessages,
    model: 'gpt-4',
    n: 1,
    user: userId,
    temperature: 0.2
  });

  storage[userId] = currentTasksResponse.choices[0].message.content

  console.log(currentTasksResponse.choices);

  res.status(200).json(chatCompletion)
})

app.listen(port, () => {
  console.log(`Briefcase app listening on port ${port}`)
})