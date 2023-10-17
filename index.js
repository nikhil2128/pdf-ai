import { pinecone } from './pinecone-client.js'
import express from "express";
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from './config/pinecone.js'
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { makeChain } from './makechain.js'

const app = express()
const port = 3000

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
      chat_history: history || [],
    });

    console.log('response', response);
    res.status(200).json(response);
  } catch (error) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
})

app.listen(port, () => {
  console.log(`Briefcase app listening on port ${port}`)
})