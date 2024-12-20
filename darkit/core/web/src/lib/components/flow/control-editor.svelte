<script lang="ts" module>
	import type { HTMLAttributes } from 'svelte/elements';
	import { type Operator, OperatorDetail } from '$lib/flow';

	interface Props extends HTMLAttributes<HTMLDivElement> {
		type: Operator;
		params?: [string | undefined, string | undefined];
		apply?: (type?: string, params?: string) => void;
	}
</script>

<script lang="ts">
	import { untrack } from 'svelte';
	import * as Card from '$lib/components/ui/card/index.js';
	import Select from '../select.svelte';
	import Input from '../input.svelte';
	import Button from '../ui/button/button.svelte';
	import { cn } from '$lib/utils';

	let fType = $state<string | undefined>();
	let fparams = $state<string>('');

	let { type: name, class: className, apply, params, ...restProps }: Props = $props();

	let detail = $derived(OperatorDetail[name]);

	$effect(() => {
		if (params) {
			const [type, parm] = params;
			untrack(() => {
				fType = type ?? detail.func[0]?.value;
				fparams = parm ?? '';
			});
		} else {
			if (detail.func.length > 0) {
				untrack(() => {
					fType = detail.func[0].value;
				});
			}
		}
	});
</script>

<Card.Root class={cn('w-72', className)} {...restProps}>
	<Card.Header>
		<Card.Title>{detail.name} Editor</Card.Title>
		<Card.Description>
			{detail.desc}
		</Card.Description>
	</Card.Header>
	<Card.Content class="flex flex-col gap-2">
		{#if detail.func.length > 1}
			<Select label="Function" bind:value={fType} options={detail.func} variant="row" />
		{/if}
		{#if detail.params}
			<Input
				bind:value={fparams}
				label="Input"
				variant="row"
				placeholder="4, 2, 9"
				tooltip="parameters for funtion. py style For example: '4, 2, 9' or '4, stride=2'"
			/>
		{/if}
	</Card.Content>
	<Card.Footer>
		<Button size="lg" class="w-full" onclick={() => apply?.(fType, fparams)}>Apply</Button>
	</Card.Footer>
</Card.Root>
