# n8n-nodes-bedrock-advanced

An advanced version of the official [n8n AWS Bedrock Chat Model node](https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.lmchatawsbedrock/), adding **prompt caching** and **detailed token usage metrics** including cache hit/miss reporting.

## Why this node?

The official n8n Bedrock node does not support [AWS Bedrock's prompt caching feature](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html). When building AI agents with long system prompts or repeated tool definitions, prompt caching can significantly reduce both latency and cost by reusing previously processed tokens across requests.

This node is a drop-in replacement that adds:

- **Prompt caching** with configurable TTL
- **Token usage metrics** including cache read/write breakdowns visible in n8n's execution data

## Features

### Prompt Caching

Enable prompt caching to automatically inject cache points into system messages before they reach the Bedrock Converse API. Bedrock will cache the system prompt prefix and reuse it across subsequent requests within the TTL window.

**Supported TTL options:**

| TTL | Supported Models |
|-----|-----------------|
| 5 minutes | All cacheable models (Claude 3.5 Sonnet v2, 3.7 Sonnet, Amazon Nova, etc.) |
| 1 hour | Claude 4.5 Opus, Sonnet 4.5, Haiku 4.5 |

### Token Usage Metrics

Every execution reports detailed token usage in the node's output data:

```json
{
  "tokenUsage": {
    "inputTokens": 150,
    "outputTokens": 500,
    "totalTokens": 650,
    "cacheReadInputTokens": 120,
    "cacheWriteInputTokens": 0
  }
}
```

| Field | Description |
|-------|-------------|
| `inputTokens` | Total input tokens processed |
| `outputTokens` | Tokens generated in the response |
| `totalTokens` | Combined input + output tokens |
| `cacheReadInputTokens` | Tokens served from cache (cache hit) |
| `cacheWriteInputTokens` | Tokens written to cache (cache miss / first call) |

Metrics work for both streaming and non-streaming invocations.

### All Original Features

Everything from the official node is preserved:

- On-demand models and inference profiles
- Dynamic model listing from your AWS account
- Temperature and max tokens configuration
- AWS credential support (access key, secret key, session token)

## Installation

### In n8n (community node)

1. Go to **Settings > Community Nodes**
2. Enter `n8n-nodes-bedrock-advanced`
3. Click **Install**

### Manual installation

```bash
cd ~/.n8n
npm install n8n-nodes-bedrock-advanced
```

Then restart n8n.

## Configuration

1. Add the **AWS Bedrock Chat Model (Advanced)** node to your workflow
2. Select your AWS credentials
3. Choose your model (on-demand or inference profile)
4. Under **Options**:
   - Toggle **Prompt Caching** to enable
   - Select **Cache TTL** (5 minutes or 1 hour)
   - Adjust temperature and max tokens as needed

## How Prompt Caching Works

When enabled, the node automatically appends a `cachePoint` marker after the system message content before sending the request to Bedrock. On the first call, Bedrock caches the system prompt tokens (`cacheWriteInputTokens` > 0). On subsequent calls within the TTL window, Bedrock reuses the cached tokens (`cacheReadInputTokens` > 0), reducing processing time and cost.

**Requirements:**
- The cached content must remain identical across requests for cache hits
- Minimum token thresholds apply per model (1,024 for Sonnet/Nova, 4,096 for Opus/Haiku)
- Maximum 4 cache checkpoints per request (this node uses 1 for the system prompt)

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Watch mode
npm run dev
```

## License

MIT
