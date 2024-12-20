<script lang="ts" module>
	import { cn } from '$lib/utils';
	import type { NodeProps } from '@xyflow/svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	type BaseProps = NodeProps & Partial<HTMLAttributes<HTMLDivElement>>;

	export const NODE_COLORS = {
		green: 'bg-green-400/80 border-green-600',
		red: 'bg-red-400/80 border-red-800',
		yellow: 'bg-yellow-400/80 border-yellow-600',
		blue: 'bg-blue-400/80 border-blue-600',
		indigo: 'bg-indigo-400/80 border-indigo-600',
		purple: 'bg-purple-400/80 border-purple-600',
		pink: 'bg-pink-400/80 border-pink-600',
		lime: 'bg-lime-500 border-lime-700',
		teal: 'bg-teal-500 border-teal-700',
		cyan: 'bg-cyan-400/80 border-cyan-600',
		amber: 'bg-amber-400/80 border-amber-600',
		violet: 'bg-violet-400/80 border-violet-600',
		orange: 'bg-orange-400/80 border-orange-600',
		brown: 'bg-brown-400/80 border-brown-600'
	};

	export interface Props extends BaseProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { Handle, Position } from '@xyflow/svelte';

	let { id, data, selected, class: className, children, ...restProps }: Props = $props();
	let color = data?.color as keyof typeof NODE_COLORS;
	let colorClass = $derived(NODE_COLORS[color] ?? NODE_COLORS.red);

	let label = $derived(data?.label ?? id) as string;
	let lastLabel = $derived.by(() => {
		const labels = label.split('.');
		return labels[labels.length - 1];
	});
</script>

<div
	class={cn(
		' hover:shadow-hover w-40 rounded border-2 p-2 text-center',
		{
			ring: selected
		},
		className,
		colorClass
	)}
	{...restProps}
>
	<Handle type="target" position={Position.Top} />
	<Handle type="source" position={Position.Bottom} />
	{#if children}
		{@render children()}
	{:else}
		<span>{lastLabel}</span>
	{/if}
</div>
