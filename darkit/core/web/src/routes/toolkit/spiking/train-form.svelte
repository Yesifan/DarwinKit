<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { zodClient } from 'sveltekit-superforms/adapters';
	import { type SuperValidated, type Infer, superForm } from 'sveltekit-superforms';
	import { Button, buttonVariants } from '$lib/components/ui/button';
	import * as Form from '$lib/components/ui/form/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { trainSpikingModel } from '$lib/apis';
	import { trainFormSchema, type TrainFormSchema } from './schema';
	import * as Select from '$lib/components/ui/select';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as m from '$lib/paraglide/messages';
	import { cn } from '$lib/utils';
	import { goto } from '$app/navigation';

	interface Props {
		data: SuperValidated<Infer<TrainFormSchema>>;
		cname: string;
		sconf: any;
		class?: string;
	}

	let { data, class: className, cname, sconf }: Props = $props();

	const form = superForm(data, {
		validators: zodClient(trainFormSchema),
		applyAction: false,
		invalidateAll: false
	});

	const { form: formData, enhance, validateForm } = form;

	const onSubmit = async (e: Event) => {
		e.preventDefault();
		const valid = await validateForm();
		if (valid.valid) {
			const [status, result] = await trainSpikingModel(cname, $formData, sconf);
			if (status === 200) {
				goto(`/toolkit/visual/${$formData.name}`);
				toast.success('Model trained successfully.');
			} else {
				toast.error(`Failed to train the model: ${result.detail}`);
			}
		}
	};
</script>

<Dialog.Root>
	<Dialog.Trigger class={cn(buttonVariants({ size: 'lg' }), className)}>{m.train()}</Dialog.Trigger>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Train the Spiking ANN Model.</Dialog.Title>
			<Dialog.Description>Fill in the training parameters and train the model.</Dialog.Description>
		</Dialog.Header>
		<form method="POST" use:enhance>
			<Form.Field {form} name="name">
				<Form.Control>
					{#snippet children({ props })}
						<Form.Label>{m.modelName()}</Form.Label>
						<Input {...props} bind:value={$formData.name} placeholder="MyModel" />
					{/snippet}
				</Form.Control>
				<Form.Description>This is the model class name in your code.</Form.Description>
				<Form.FieldErrors />
			</Form.Field>
			<Form.Field {form} name="device">
				<Form.Control>
					{#snippet children({ props })}
						<Form.Label>device</Form.Label>
						<Select.Root {...props} type="single" bind:value={$formData.device}>
							<Select.Trigger>
								{$formData.device ?? 'Select a device'}
							</Select.Trigger>
							<Select.Content>
								<Select.Item value="cpu" label="CPU">CPU</Select.Item>
								<Select.Item value="cuda" label="CUDA">CUDA</Select.Item>
							</Select.Content>
						</Select.Root>
					{/snippet}
				</Form.Control>
				<Form.FieldErrors />
			</Form.Field>
			<Form.Field {form} name="max_step">
				<Form.Control>
					{#snippet children({ props })}
						<Form.Label>max_step</Form.Label>
						<Input {...props} bind:value={$formData.max_step} placeholder="1000" />
					{/snippet}
				</Form.Control>
				<Form.FieldErrors />
			</Form.Field>
			<Form.Field {form} name="batch_size">
				<Form.Control>
					{#snippet children({ props })}
						<Form.Label>batch_size</Form.Label>
						<Input {...props} bind:value={$formData.batch_size} placeholder="1" />
					{/snippet}
				</Form.Control>
				<Form.FieldErrors />
			</Form.Field>
			<Form.Field {form} name="T">
				<Form.Control>
					{#snippet children({ props: { id, ...props } })}
						<Form.Label>T</Form.Label>
						<Input {...props} bind:value={$formData.T} type="number" />
					{/snippet}
				</Form.Control>
				<Form.Description>时间步在 SNN 中是模拟连续时间动态的重要基础单位。</Form.Description>
				<Form.FieldErrors />
			</Form.Field>
			<Form.Field {form} name="learning_rate">
				<Form.Control>
					{#snippet children({ props })}
						<Form.Label>learning_rate</Form.Label>
						<Input {...props} bind:value={$formData.learning_rate} placeholder="0.01" />
					{/snippet}
				</Form.Control>
				<Form.FieldErrors />
			</Form.Field>
			<Form.Field {form} name="save_step_interval">
				<Form.Control>
					{#snippet children({ props })}
						<Form.Label>save_step_interval</Form.Label>
						<Input {...props} bind:value={$formData.save_step_interval} placeholder="100" />
					{/snippet}
				</Form.Control>
				<Form.FieldErrors />
			</Form.Field>
		</form>
		<Dialog.Footer>
			<Button type="submit" onclick={onSubmit}>{m.submit()}</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
