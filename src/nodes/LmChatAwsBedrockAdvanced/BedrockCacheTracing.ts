import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { Serialized } from '@langchain/core/load/serializable';
import type { LLMResult } from '@langchain/core/outputs';
import {
	NodeConnectionTypes,
	type ISupplyDataFunctions,
} from 'n8n-workflow';

/**
 * Lightweight callback that surfaces prompt caching metrics
 * in the AI Agent's Logs panel as a separate log entry.
 * Runs alongside N8nLlmTracing without conflicting.
 */
export class BedrockCacheTracing extends BaseCallbackHandler {
	name = 'BedrockCacheTracing';

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
			[[{ json: { action: 'cacheMetrics' } }]],
		);
		const index = typeof result === 'number' ? result : (result as any).index ?? 0;
		this.runMap.set(runId, { index });
	}

	override async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
		const runDetails = this.runMap.get(runId);
		if (!runDetails) return;
		this.runMap.delete(runId);

		const metrics = this.extractCacheMetrics(output);

		this.executionFunctions.addOutputData(
			NodeConnectionTypes.AiLanguageModel,
			runDetails.index,
			[[
				{
					json: {
						promptCaching: metrics,
					},
				},
			]],
		);
	}

	override async handleLLMError(
		_error: Error,
		runId: string,
	): Promise<void> {
		this.runMap.delete(runId);
	}

	private extractCacheMetrics(output: LLMResult): Record<string, unknown> {
		const firstGen = output.generations?.[0]?.[0];
		if (!firstGen) return { status: 'unknown', error: 'no generation' };

		const message = (firstGen as any).message;
		if (!message) return { status: 'unknown', error: 'no message' };

		// Check if our CachingChatBedrockConverse already attached metrics
		const existing = message.response_metadata?.promptCachingMetrics;
		if (existing) return existing;

		// Fallback: try to extract from raw usage
		const rawUsage = message.response_metadata?.usage
			?? message.response_metadata?.metadata?.usage;

		if (!rawUsage) return { status: 'unknown', error: 'no usage data' };

		const cacheRead = rawUsage.cacheReadInputTokens || 0;
		const cacheWrite = rawUsage.cacheWriteInputTokens || 0;

		let status = 'NO CACHE';
		if (cacheRead > 0) status = 'CACHE HIT';
		else if (cacheWrite > 0) status = 'CACHE WRITTEN';

		return {
			status,
			tokensReadFromCache: cacheRead,
			tokensWrittenToCache: cacheWrite,
		};
	}
}
