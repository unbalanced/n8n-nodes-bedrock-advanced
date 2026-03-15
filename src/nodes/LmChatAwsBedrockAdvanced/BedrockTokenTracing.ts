import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { Serialized } from '@langchain/core/load/serializable';
import type { LLMResult } from '@langchain/core/outputs';
import type { AIMessage } from '@langchain/core/messages';
import {
	NodeConnectionTypes,
	type ISupplyDataFunctions,
} from 'n8n-workflow';

export interface BedrockTokenUsage {
	inputTokens: number;
	outputTokens: number;
	totalTokens: number;
	cacheReadInputTokens: number;
	cacheWriteInputTokens: number;
}

/**
 * Custom LangChain callback handler that captures Bedrock token usage
 * including prompt caching metrics, and surfaces them in n8n's execution data.
 */
export class BedrockTokenTracing extends BaseCallbackHandler {
	name = 'BedrockTokenTracing';

	private runMap = new Map<string, { index: number }>();

	constructor(private executionFunctions: ISupplyDataFunctions) {
		super();
	}

	override async handleLLMStart(
		_llm: Serialized,
		_prompts: string[],
		runId: string,
	): Promise<void> {
		const result = this.executionFunctions.addInputData(
			NodeConnectionTypes.AiLanguageModel,
			[[{ json: { action: 'llmCall', prompts: _prompts } }]],
		);
		const index = typeof result === 'number' ? result : (result as any).index ?? 0;
		this.runMap.set(runId, { index });
	}

	override async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
		const runDetails = this.runMap.get(runId);
		if (!runDetails) return;
		this.runMap.delete(runId);

		const usage = this.extractUsage(output);

		this.executionFunctions.addOutputData(
			NodeConnectionTypes.AiLanguageModel,
			runDetails.index,
			[[
				{
					json: {
						tokenUsage: usage,
					},
				},
			]],
		);
	}

	override async handleLLMError(
		_error: Error,
		runId: string,
	): Promise<void> {
		const runDetails = this.runMap.get(runId);
		if (!runDetails) return;
		this.runMap.delete(runId);
	}

	/**
	 * Debug codes encoded in cacheWriteInputTokens when no real value found:
	 *  -1 = no generation in output
	 *  -2 = no message on generation
	 *  -3 = no response_metadata on message
	 *  -4 = no usage object found at any path
	 *  -N = usage object found with N keys but no cache fields (e.g. -3 means 3 keys)
	 *       encoded as -(100 + keyCount)
	 */
	private extractUsage(output: LLMResult): BedrockTokenUsage {
		const usage: BedrockTokenUsage = {
			inputTokens: 0,
			outputTokens: 0,
			totalTokens: 0,
			cacheReadInputTokens: 0,
			cacheWriteInputTokens: -1,
		};

		const firstGen = output.generations?.[0]?.[0];
		if (!firstGen) return usage;

		const message = (firstGen as any).message as AIMessage | undefined;
		if (!message) { usage.cacheWriteInputTokens = -2; return usage; }

		// Standard usage from usage_metadata (set by @langchain/aws)
		const usageMeta = message.usage_metadata;
		if (usageMeta) {
			usage.inputTokens = usageMeta.input_tokens ?? 0;
			usage.outputTokens = usageMeta.output_tokens ?? 0;
			usage.totalTokens = usageMeta.total_tokens ?? 0;
		}

		const responseMeta = (message as any).response_metadata;
		if (!responseMeta) { usage.cacheWriteInputTokens = -3; return usage; }

		// Check all possible usage object locations
		const rawUsage = responseMeta.usage
			?? responseMeta.metadata?.usage
			?? responseMeta.metrics?.usage;

		if (!rawUsage) { usage.cacheWriteInputTokens = -4; return usage; }

		const usageKeys = Object.keys(rawUsage);

		// Try ALL known field name variations for cache metrics
		const cacheRead =
			rawUsage.cacheReadInputTokenCount
			?? rawUsage.cacheReadInputTokens
			?? rawUsage.CacheReadInputTokenCount
			?? rawUsage.CacheReadInputTokens
			?? rawUsage.cache_read_input_tokens
			?? rawUsage.cache_read_input_token_count;

		const cacheWrite =
			rawUsage.cacheWriteInputTokenCount
			?? rawUsage.cacheWriteInputTokens
			?? rawUsage.CacheWriteInputTokenCount
			?? rawUsage.CacheWriteInputTokens
			?? rawUsage.cacheCreationInputTokenCount
			?? rawUsage.cacheCreationInputTokens
			?? rawUsage.cache_write_input_tokens
			?? rawUsage.cache_creation_input_tokens;

		if (cacheRead !== undefined || cacheWrite !== undefined) {
			usage.cacheReadInputTokens = cacheRead ?? 0;
			usage.cacheWriteInputTokens = cacheWrite ?? 0;
		} else {
			// No cache fields found — encode key count as debug signal
			// e.g. -103 means usage object has 3 keys but none are cache fields
			usage.cacheWriteInputTokens = -(100 + usageKeys.length);
			// Encode first key's char code in cacheRead as additional debug
			usage.cacheReadInputTokens = usageKeys.length > 0
				? -(usageKeys[0].charCodeAt(0))
				: 0;
		}

		if (!usage.inputTokens) {
			usage.inputTokens = rawUsage.inputTokens ?? 0;
		}
		if (!usage.outputTokens) {
			usage.outputTokens = rawUsage.outputTokens ?? 0;
		}
		if (!usage.totalTokens) {
			usage.totalTokens = rawUsage.totalTokens ?? usage.inputTokens + usage.outputTokens;
		}

		return usage;
	}
}
