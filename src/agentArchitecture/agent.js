import {DuckDuckGoSearch} from "@langchain/community/tools/duckduckgo_search";
import {Calculator} from "@langchain/community/tools/calculator";
import {ChatGoogleGenerativeAI} from "@langchain/google-genai";
import {
  Annotation,
  messagesStateReducer,
  START,
  StateGraph
} from "@langchain/langgraph";
import {ToolNode, toolsCondition} from "@langchain/langgraph/prebuilt";
import {HumanMessage} from "@langchain/core/messages";

const search = new DuckDuckGoSearch()
const calculator = new Calculator()
const tools = [search, calculator]
const model = new ChatGoogleGenerativeAI(
    {model: "gemini-2.5-flash-lite", temperature: 0.1}).bindTools(tools)
const annotation = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer, default: () => []
  })
})

async function modelNode(state) {
  const res = await model.invoke(state.messages)
  return {messages: res}
}

const builder = new StateGraph(annotation)
.addNode('model', modelNode)
.addNode('tools', new ToolNode(tools))
.addEdge(START, 'model')
.addConditionalEdges('model', toolsCondition)
.addEdge('tools', 'model')
const graph = builder.compile()

const input = {
  messages: [
    new HumanMessage("대한민국의 제13대 대통령이 사망했을 때 몇 살이었나요?"), // 89세
  ],
}
for await (const chunk of await graph.stream(input)) {
  console.log(chunk)
}