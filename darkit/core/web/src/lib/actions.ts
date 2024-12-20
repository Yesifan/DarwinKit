import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
// 引入标题，提示框，直角坐标系，数据集，内置数据转换器组件，组件后缀都为 Component
import {
	LegendComponent,
	TooltipComponent,
	ToolboxComponent,
	GridComponent,
	DatasetComponent,
	TransformComponent,
	MarkLineComponent
} from 'echarts/components';
// 标签自动布局、全局过渡动画等特性
import { LabelLayout, UniversalTransition } from 'echarts/features';
// 引入 Canvas 渲染器，注意引入 CanvasRenderer 或者 SVGRenderer 是必须的一步
import { CanvasRenderer } from 'echarts/renderers';
import type { LineSeriesOption } from 'echarts/charts';
import type {
	// 组件类型的定义后缀都为 ComponentOption
	LegendComponentOption,
	ToolboxComponentOption,
	TooltipComponentOption,
	GridComponentOption,
	DatasetComponentOption,
	MarkLineComponentOption
} from 'echarts/components';
import type { ComposeOption } from 'echarts/core';

export type ECOption = ComposeOption<
	| LineSeriesOption
	| TooltipComponentOption
	| GridComponentOption
	| DatasetComponentOption
	| ToolboxComponentOption
	| LegendComponentOption
	| MarkLineComponentOption
>;
export const ssr = false;

echarts.use([
	ToolboxComponent,
	TooltipComponent,
	LegendComponent,
	GridComponent,
	DatasetComponent,
	TransformComponent,
	MarkLineComponent,
	LineChart,
	LabelLayout,
	UniversalTransition,
	CanvasRenderer
]);

export const chart = (node: HTMLElement, options: ECOption) => {
	const myChart = echarts.init(node);
	myChart.setOption(options);

	const resizeObserver = new ResizeObserver(() => {
		myChart.resize();
	});

	resizeObserver.observe(node);

	return {
		update(options: ECOption) {
			myChart.setOption(options, { notMerge: true });
			myChart.resize();
		},
		destroy() {
			myChart.dispose();
			resizeObserver.unobserve(node);
		}
	};
};
