<script lang="ts">
	import { cn } from '$lib/utils';
	import { chart } from '$lib/actions';
	import type { ECOption } from '$lib/actions';

	const base_options: ECOption = {
		animation: true,
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
			type: 'category', // 使用类别轴，ECharts 会自动以索引为 x 轴
			splitLine: {
				show: false
			}
		},
		yAxis: {
			id: 'loss',
			type: 'value',
			position: 'left',
			min: function (value) {
				return Math.floor(value.min) - 0.5;
			}
		}
	};

	let {
		data = [
			[10, 20, 15, 25, 30],
			[3, 4, 7, 9, 10]
		],
		splits = [],
		className
	}: { data?: number[][]; splits?: number[]; className?: string } = $props();

	let options: ECOption = $derived.by(() => {
		return {
			...base_options,
			series: data.map((d, i) => ({
				data: d,
				type: 'line',
				smooth: true, // 使折线平滑
				showSymbol: false, // 隐藏折线上的点
				markLine:
					i === 0
						? {
								symbol: 'none', // 隐藏箭头符号
								label: { show: false }, // 隐藏分割线的标签
								lineStyle: {
									color: 'red', // 分割线颜色
									type: 'dashed', // 分割线类型
									width: 2 // 分割线宽度
								},
								data: splits.map((split) => ({ xAxis: split }))
							}
						: undefined
			}))
		};
	});

	export { className as class };
</script>

<div use:chart={options} class={cn(className, 'h-[800px] w-full')}></div>
