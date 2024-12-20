<script lang="ts">
	import { cn } from '$lib/utils';
	import { Expand } from 'lucide-svelte';
	import Button from '../ui/button/button.svelte';
	import BaseNode, { type Props as BaseProps } from './base-node.svelte';

	type OnClick = (id: BaseProps['id']) => void;

	let { id, data, selected, class: className, ...restProps }: BaseProps = $props();
	let onclick = $derived(data.onClick) as OnClick;

	let label = $derived(data?.label ?? id) as string;
	let lastLabel = $derived.by(() => {
		const labels = label.split('.');
		return labels[labels.length - 1];
	});
</script>

<BaseNode class={cn(className)} {id} {data} {selected} {...restProps}>
	<span>{lastLabel}</span>
	{#if data.sub_module_size}
		<span
			class="bg-primary/30 absolute right-2 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-sm"
		>
			{data.sub_module_size}
		</span>
	{/if}

	{#if selected}
		<Button
			size="icon"
			variant="outline"
			class="absolute right-0 top-1/2 size-8 -translate-y-1/2 translate-x-10 rounded-full"
			onclick={() => onclick?.(id)}
		>
			<Expand class="h-4 w-4" />
		</Button>
	{/if}
</BaseNode>
