<script lang="ts">
	import { Upload } from 'lucide-svelte';
	import { toast } from 'svelte-sonner';
	import * as m from '$lib/paraglide/messages';
	import Input from '$lib/components/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Button, buttonVariants } from '$lib/components/ui/button';
	import { post } from '../api';
	import { cn } from '$lib/utils';

	let loading = $state(false);
	let modelName: string | null = $state(null);
	let modelPackageFile: FileList | null | undefined = $state(null);
	let modelFileName = $derived((modelPackageFile?.[0] as any).name);

	const uploadWeight = async () => {
		loading = true;
		const formData = new FormData();
		formData.append('name', modelName!);
		formData.append('file', modelPackageFile?.[0] as any);
		const res = await post('/model/upload/weight', formData);
		loading = false;
		if (res.status === 200) {
			modelName = null;
			modelPackageFile = null;
			toast.success('Model weight uploaded successfully');
		} else {
			const result = await res.json();
			toast.error(result.detail);
		}
	};
</script>

<div class="flex h-full flex-1 flex-col gap-4 overflow-x-hidden p-2">
	<section class="flex flex-col items-center gap-4 py-2">
		<h2 class="text-2xl font-semibold tracking-tight">{m.upload_model()}</h2>
		<p class="border-l-2 pl-6">
			{m.upload_model_quote()}
		</p>
		<Input value="" label={m.modelName()} />
		<div class="flex w-96 flex-col gap-1.5">
			<Label>{m.model_file()}</Label>
			<label
				for="upload-model-package"
				class={cn(buttonVariants({ variant: 'outline' }), 'cursor-pointer')}
			>
				<Upload size={18} class="mr-2" />
				{m.upload_model()}
			</label>
			<input type="file" id="upload-model-package" class="hidden" accept=".gz" />
		</div>
		<Button class="mt-4 w-96" disabled>{m.submit()}</Button>
	</section>
	<section class="flex flex-col items-center gap-4 py-2">
		<h2 class="text-2xl font-semibold tracking-tight">{m.upload_model_weight()}</h2>
		<p class="border-l-2 pl-6">
			{m.upload_model_weight_quote()}
		</p>
		<Input bind:value={modelName} label={m.modelName()} />
		<div class="flex w-96 flex-col gap-1.5">
			<Label>{m.model_weight_file()}</Label>
			<label
				for="upload-weight-package"
				class={cn(buttonVariants({ variant: 'outline' }), 'cursor-pointer')}
			>
				<Upload size={18} class="mr-2" />
				{modelFileName ?? m.upload_model_weight()}
			</label>
			<input
				type="file"
				class="hidden"
				id="upload-weight-package"
				accept=".gz"
				bind:files={modelPackageFile}
			/>
		</div>
		<Button
			class="mt-4 w-96"
			disabled={!modelFileName || !modelName || loading}
			onclick={uploadWeight}>{m.submit()}</Button
		>
	</section>
</div>
