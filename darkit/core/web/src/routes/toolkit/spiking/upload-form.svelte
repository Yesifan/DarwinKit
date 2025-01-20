<script lang="ts">
	import type { NXEdge, NXNode } from '$lib/components/model-editor.svelte';
	import { buttonVariants } from '$lib/components/ui/button';
	import * as Form from '$lib/components/ui/form/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { cn } from '$lib/utils';
	import { Upload } from 'lucide-svelte';
	import { uploadFormSchema, type UploadFormSchema } from './schema';
	import { type SuperValidated, type Infer, superForm, fileProxy } from 'sveltekit-superforms';
	import { zodClient } from 'sveltekit-superforms/adapters';
	import { uploadSpikingModel } from '$lib/apis';
	import * as m from '$lib/paraglide/messages';
	import { toast } from 'svelte-sonner';

	interface Props {
		data: SuperValidated<Infer<UploadFormSchema>>;
		cname: string;
		graph?: [NXNode[], NXEdge[]] | null;
		onupdate?: (nodes: NXNode[], edges: NXEdge[]) => void;
		class?: string;
	}

	let {
		data,
		cname = $bindable(''),
		graph = $bindable(null),
		onupdate,
		class: className
	}: Props = $props();

	const form = superForm(data, {
		validators: zodClient(uploadFormSchema),
		applyAction: false,
		invalidateAll: false
	});

	const { form: formData, enhance, validateForm } = form;
	const file = fileProxy(form, 'file');

	const onSubmit = async (e: Event) => {
		e.preventDefault();
		const valid = await validateForm();
		if (valid.valid) {
			try {
				const { model_name, file } = $formData;
				const res = await uploadSpikingModel(model_name, file);
				cname = model_name;
				graph = res;
				onupdate?.(res[0], res[1]);
			} catch (e) {
				console.error((e as any).detail);
			}
		} else {
			toast.warning('Please fill the form correctly');
		}
	};
</script>

<form method="POST" use:enhance class={className} enctype="multipart/form-data">
	<Form.Field {form} name="model_name">
		<Form.Control>
			{#snippet children({ props })}
				<Form.Label>{m.modelName()}</Form.Label>
				<Input {...props} bind:value={$formData.model_name} placeholder="MyModel" />
			{/snippet}
		</Form.Control>
		<Form.Description>This is the model class name in you code.</Form.Description>
		<Form.FieldErrors />
	</Form.Field>
	<Form.Field {form} name="file">
		<Form.Control>
			{#snippet children({ props: { id, ...props } })}
				<Form.Label>{m.modelFile()}</Form.Label>
				<div>
					<label
						for={id}
						class={cn(buttonVariants({ variant: 'outline', size: 'full' }), 'cursor-pointer')}
					>
						<Upload size={18} class="mr-2" />
						{#if $formData.file}
							<span class="text-muted-foreground">{$formData.file.name}</span>
						{:else}
							<span>{m.upload_model()}</span>
						{/if}
					</label>
					<input
						{id}
						type="file"
						class="hidden"
						accept=".py"
						multiple={false}
						{...props}
						bind:files={$file}
					/>
				</div>
			{/snippet}
		</Form.Control>
		<Form.Description>This is a python file with you model code.</Form.Description>
		<Form.FieldErrors />
	</Form.Field>
	<Form.Button size="full" class="my-4" onclick={onSubmit}>{m.submit()}</Form.Button>
</form>
