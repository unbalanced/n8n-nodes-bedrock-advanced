const esbuild = require('esbuild');
const { cpSync } = require('fs');

const entryPoints = [
	'src/nodes/LmChatAwsBedrockAdvanced/LmChatAwsBedrockAdvanced.node.ts',
	'src/nodes/LmChatBedrockClaude/LmChatBedrockClaude.node.ts',
	'src/index.ts',
];

esbuild.buildSync({
	entryPoints,
	bundle: true,
	platform: 'node',
	target: 'node20',
	outdir: 'dist',
	format: 'cjs',
	sourcemap: true,
	outbase: 'src',
	// Only n8n-workflow is external - provided by n8n at runtime
	// Everything else is bundled to avoid shallow install issues
	external: ['n8n-workflow'],
});

// Copy SVG icons
cpSync(
	'src/nodes/LmChatAwsBedrockAdvanced/bedrock.svg',
	'dist/nodes/LmChatAwsBedrockAdvanced/bedrock.svg',
);
cpSync(
	'src/nodes/LmChatBedrockClaude/bedrock-claude.svg',
	'dist/nodes/LmChatBedrockClaude/bedrock-claude.svg',
);

console.log('Build complete');
