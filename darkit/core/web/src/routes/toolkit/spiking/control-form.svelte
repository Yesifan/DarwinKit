<script lang="ts">
	import * as Form from '$lib/components/ui/form/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { controlFormSchema, type ControlFormSchema, Modules } from './schema';
	import { type SuperValidated, type Infer, superForm } from 'sveltekit-superforms';
	import { zodClient } from 'sveltekit-superforms/adapters';
	import * as Select from '$lib/components/ui/select';
	import { cn } from '$lib/utils';
	import * as m from '$lib/paraglide/messages';
	import * as Card from '$lib/components/ui/card';

	interface Props {
		data: SuperValidated<Infer<ControlFormSchema>>;
		id: string;
		class?: string;
		onupdate?: (id: string, formdata: Infer<ControlFormSchema>) => void;
	}

	let { data, id, onupdate, class: className }: Props = $props();

	const form = superForm(data, {
		validators: zodClient(controlFormSchema),
		applyAction: false,
		invalidateAll: false
	});

	const { form: formData, enhance, validateForm } = form;

	const onSubmit = async (e: Event) => {
		e.preventDefault();
		const valid = await validateForm();
		if (valid.valid) {
			onupdate?.(id, $formData);
		}
	};
</script>

<Card.Root class={className}>
	<form method="POST" use:enhance>
		<Card.Header>
			<Card.Title>{id} Editor</Card.Title>
		</Card.Header>
		<Card.Content class="flex flex-col gap-2">
			<Form.Field {form} name="type">
				<Form.Control>
					{#snippet children({ props })}
						<Form.Label>Spiking Module</Form.Label>
						<Select.Root {...props} type="single" bind:value={$formData.type}>
							<Select.Trigger>
								{$formData.type ?? 'Select a spiking module'}
							</Select.Trigger>
							<Select.Content>
								{#each Modules as item}
									<Select.Item value={item.value} label={item.label}>
										{item.label}
									</Select.Item>
								{/each}
							</Select.Content>
						</Select.Root>
					{/snippet}
				</Form.Control>
				<Form.Description>选择你想要替换的 Spiking Module</Form.Description>
				<Form.FieldErrors />
			</Form.Field>
			<Form.Field {form} name="v_threshold">
				<Form.Control>
					{#snippet children({ props: { id, ...props } })}
						<Form.Label>v_threshold</Form.Label>
						<Input {...props} bind:value={$formData.v_threshold} type="number" />
					{/snippet}
				</Form.Control>
				<Form.Description>时间步在 SNN 中是模拟连续时间动态的重要基础单位。</Form.Description>
				<Form.FieldErrors />
			</Form.Field>
		</Card.Content>
		<Card.Footer>
			<Form.Button size="full" class="my-4" onclick={onSubmit}>{m.apply()}</Form.Button>
		</Card.Footer>
	</form>
</Card.Root>
