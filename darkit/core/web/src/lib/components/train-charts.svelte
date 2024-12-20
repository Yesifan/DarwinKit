<script lang="ts">
	import { cn } from '$lib/utils';
	import { chart } from '$lib/actions';
	import type { ECOption } from '$lib/actions';
	import type { LineSeriesOption } from 'echarts/charts';
	import type { LegendComponentOption } from 'echarts/components';

	let options: ECOption = {
		animation: false,
		grid: {
			top: 100,
			left: 4,
			right: 4,
			bottom: 4,
			containLabel: true
		},
		toolbox: {
			feature: {
				saveAsImage: {}
			}
		},
		tooltip: {
			trigger: 'axis'
		},
		xAxis: {
			type: 'value',
			splitLine: {
				show: false
			}
		},
		yAxis: [
			{
				id: 'loss',
				type: 'value',
				position: 'left',
				min: function (value) {
					return Math.floor(value.min) - 0.5;
				}
			},
			{
				id: 'ppl',
				type: 'value',
				position: 'right'
			}
		],
		series: [
			{
				data: [
					[1, 2],
					[3, 4]
				],
				type: 'line'
			}
		]
	};

	let className: string | undefined = undefined;
	export { className as class };
	const SERIES_GROUP = {
		train_loss: ['total_tokens', 'train_loss'],
		val_loss: ['total_tokens', 'val_loss'],
		val_ppl: ['total_tokens', 'val_ppl']
	};
	let seriesGroup: { [key: string]: { [key: string]: any } } = {};

	export { seriesGroup as series };
	export let datas: { [key: string]: any[] } = {};

	$: series = Object.entries(datas).flatMap(([model_name, val]) => {
		return Object.entries(seriesGroup[model_name] ?? SERIES_GROUP).flatMap(([title, series]) => {
			const isArray = Array.isArray(series);
			const seriesType = isArray ? 'line' : series.type;
			const [x, y] = isArray ? series : series.fields;
			const data = val.filter((val) => !!val[y]).map((val) => [Number(val[x]), Number(val[y])]);
			return {
				name: `${model_name}[${title}]`,
				data: data,
				type: seriesType,
				smooth: true,
				showSymbol: title.includes('val'),
				yAxisIndex: title.includes('loss') ? 0 : 1
			} as LineSeriesOption;
		});
	});
	$: legend = { data: series.map((s) => s.name), show: true, top: 20 } as LegendComponentOption;
	$: options = { ...options, series, legend };
</script>

<div use:chart={options} class={cn(className, 'h-[800px] w-full')}></div>
