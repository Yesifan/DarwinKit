<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { Trash2 } from 'lucide-svelte';
	import { invalidateAll } from '$app/navigation';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
	import Button, { buttonVariants } from '$lib/components/ui/button/button.svelte';
	import { delete_ } from '../api';
	import * as m from '$lib/paraglide/messages';

	let { data } = $props();

	let selectModels: string[] = $state([]);

	async function delModels(models: string[]) {
		if (models.length === 0) return;
		try {
			const response = await delete_('/models/', { check_models: models });
			if (response.status === 200) {
				toast.success('Models deleted successfully');
				invalidateAll();
			} else {
				toast.error('Failed to delete models:');
			}
		} catch (error) {
			console.error('Failed to deleting models:', error);
		} finally {
			models.length = 0; // 清空选中的模型
		}
	}
</script>

<div class="h-full flex-1 overflow-x-hidden p-8">
	<h2 class="scroll-m-20 pb-4 text-3xl font-semibold tracking-tight transition-colors first:mt-0">
		{m.modelVisualize()}
	</h2>
	<div class="pb-4 text-xl">{m.modelVisualHead()}</div>
	<div class="flex flex-wrap gap-2">
		<ToggleGroup.Root
			type="multiple"
			variant="outline"
			class="flex-wrap justify-start"
			bind:value={selectModels}
		>
			{#each data.networks as model}
				<ToggleGroup.Item value={model.name}>
					{#if model.config.fork}
						[{model.config.fork}] {model.name}
					{:else}
						{model.name}
					{/if}
				</ToggleGroup.Item>
			{:else}
				<div>EMPTY</div>
			{/each}
		</ToggleGroup.Root>
	</div>

	<div class="mt-16 flex gap-8">
		<Button
			size="lg"
			class="flex-1"
			disabled={selectModels.length === 0}
			href={`/lm/visual/${selectModels.join('&')}`}
		>
			{m.showDetails()}
		</Button>

		<Dialog.Root>
			<Dialog.Trigger
				disabled={selectModels.length === 0}
				class={buttonVariants({ variant: 'destructive', size: 'lg' })}
			>
				<Trash2 class="mr-2" size="18" />
				{m.mDeleteModel()}
			</Dialog.Trigger>
			<Dialog.Content>
				<Dialog.Header>
					<Dialog.Title>Are you absolutely sure?</Dialog.Title>
					<Dialog.Description>
						This action cannot be undone. This will permanently delete the selected models.
					</Dialog.Description>
				</Dialog.Header>
				<Dialog.Footer>
					<Dialog.Close onclick={() => delModels(selectModels)}>Ok</Dialog.Close>
				</Dialog.Footer>
			</Dialog.Content>
		</Dialog.Root>
	</div>
</div>
