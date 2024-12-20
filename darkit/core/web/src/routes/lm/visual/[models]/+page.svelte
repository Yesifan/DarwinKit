<script lang="ts" module>
	import type { ModelDetail } from '$lib/apis/lm';
</script>

<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onDestroy, onMount } from 'svelte';
	import { ReconnectingWebSocket } from '$lib';
	import { get, post } from '$lib/api';
	import * as Card from '$lib/components/ui/card';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
	import Button from '$lib/components/ui/button/button.svelte';
	import TrainCharts from '$lib/components/train-charts.svelte';
	import * as m from '$lib/paraglide/messages';
	import Code from '$lib/components/code.svelte';

	let { data } = $props();
	let destried = $state(false);
	let training = $state(false);
	let wss: ReconnectingWebSocket[] = $state([]);
	let errors: { name: string; detail: string }[] = $state([]);
	let modelInfos: ModelDetail[] = $state([]);
	let selectModels: string[] = $state([]);

	let modelLogDatas: { [key: string]: string[] } = $state({});
	let modelLogSeries = $derived(
		modelInfos.reduce(
			(acc, model) => {
				acc[model.model_name] = model?.external_config?.series;
				return acc;
			},
			{} as Record<string, any>
		)
	);

	const getModel = async (model: string) => {
		const res = await get(`/lm/models/${model}`);
		if (res.status == 200) {
			const data = await res.json();
			return data;
		} else if (res.status === 400) {
			const data = await res.json();
			toast.error(`${model} 训练失败: ${data.detail}`, { duration: 10000 });
			return null;
		}
		if (destried) return null;
		toast.warning('fetch failed, retrying in 5s');
		await new Promise((resolve) => setTimeout(resolve, 5000));
		return getModel(model);
	};

	const getModels = async (models: string[]) => {
		const res = await Promise.all(models.map((model) => getModel(model)));
		const filterRes = res.filter((r) => r !== null);
		return filterRes;
	};

	const addLogDataWS = async (model: string) => {
		let connect = 1;
		const wsURL = `/api/lm/models/${model}/logging`;
		const ws = new ReconnectingWebSocket(wsURL); // 使用相对路径，假设前端和后端部署在同一个域名下
		wss.push(ws);
		ws.onMessage = (data) => {
			if (data) {
				training = true;
				if (data === 'EOF') {
					training = false;
					toast.success(`${model} 训练完成`);
				} else if (data.startsWith('EXCEPTION')) {
					training = false;
					errors.push({ name: model, detail: data });
				} else {
					const data_json = JSON.parse(data);
					if (!modelLogDatas[model] || connect == 1) {
						connect = 0;
						modelLogDatas[model] = data_json;
					} else {
						modelLogDatas[model] = [...modelLogDatas[model], ...data_json];
					}
					modelLogDatas = { ...modelLogDatas };
				}
			} else {
				console.log('data is empty');
			}
		};
		ws.onClose = (event) => {
			if (event.code !== 1000) {
				// 非正常关闭，自动重连
				connect = 1;
			}
		};
	};

	async function stopTraining(model: string) {
		console.log('Stop training', model);
		const res = await post(`/lm/models/${model}/stop`);
		if (res.status === 200) {
			console.log('stop training success');
		} else {
			const data = await res.json();
			toast.error(`Stop training failed: ${data.detail}`, { duration: 10000 });
		}
	}

	onMount(async () => {
		modelInfos = await getModels(data.models);
		selectModels = modelInfos.map((model) => model.model_name);
		data.models.forEach((model) => {
			addLogDataWS(model);
		});
	});

	onDestroy(() => {
		destried = true;
		wss.forEach((ws) => {
			ws.close();
		});
	});
</script>

<div class="h-full flex-1 overflow-x-hidden p-8">
	<div class="flex justify-between">
		<h1 class="pb-4 text-3xl font-semibold">
			{m.modelVisualize()}
		</h1>
		{#if data.models.length === 1 && training}
			<Button variant="destructive" onclick={() => stopTraining(data.models[0])}>
				Stop Training
			</Button>
		{/if}
	</div>
	{#if modelInfos.length === 0}
		<div class="pb-16 text-xl">Loading...</div>
	{:else}
		<TrainCharts datas={modelLogDatas} series={modelLogSeries} />
		<h2 class="py-4 text-3xl font-semibold">Model Config</h2>
		<div class="flex flex-col gap-4">
			<ToggleGroup.Root
				type="multiple"
				variant="outline"
				class="justify-start"
				bind:value={selectModels}
			>
				{#each data.models as model}
					<ToggleGroup.Item value={model}>
						{model}
					</ToggleGroup.Item>
				{/each}
			</ToggleGroup.Root>

			<section class="flex min-h-96 flex-1 flex-wrap gap-2">
				{#each modelInfos as model}
					{#if selectModels.includes(model.model_name)}
						<Card.Root>
							<Card.Header>
								<Card.Title>{model.model_name}</Card.Title>
							</Card.Header>
							<Card.Content class="grid grid-cols-3">
								{#each Object.entries(model.model_config) as [key, value]}
									<div>
										<div class="text-primary/60 text-sm">{key}</div>
										<div class="text-primary">{value ?? 'none'}</div>
									</div>
								{/each}
							</Card.Content>
						</Card.Root>
					{/if}
				{/each}
			</section>
		</div>
	{/if}
</div>
{#each errors as { name, detail }}
	<AlertDialog.Root open={true}>
		<AlertDialog.Content class="max-w-3xl">
			<AlertDialog.Header>
				<AlertDialog.Title>{name} Error</AlertDialog.Title>
			</AlertDialog.Header>
			<AlertDialog.Description class="h-96 overflow-hidden">
				<Code lang="bash" content={detail} />
			</AlertDialog.Description>
			<AlertDialog.Footer>
				<AlertDialog.Cancel>Ok</AlertDialog.Cancel>
			</AlertDialog.Footer>
		</AlertDialog.Content>
	</AlertDialog.Root>
{/each}
