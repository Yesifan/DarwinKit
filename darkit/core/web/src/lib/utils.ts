import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { cubicOut } from 'svelte/easing';
import type { TransitionConfig } from 'svelte/transition';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

type FlyAndScaleParams = {
	y?: number;
	x?: number;
	start?: number;
	duration?: number;
};

export const flyAndScale = (
	node: Element,
	params: FlyAndScaleParams = { y: -8, x: 0, start: 0.95, duration: 150 }
): TransitionConfig => {
	const style = getComputedStyle(node);
	const transform = style.transform === 'none' ? '' : style.transform;

	const scaleConversion = (valueA: number, scaleA: [number, number], scaleB: [number, number]) => {
		const [minA, maxA] = scaleA;
		const [minB, maxB] = scaleB;

		const percentage = (valueA - minA) / (maxA - minA);
		const valueB = percentage * (maxB - minB) + minB;

		return valueB;
	};

	const styleToString = (style: Record<string, number | string | undefined>): string => {
		return Object.keys(style).reduce((str, key) => {
			if (style[key] === undefined) return str;
			return str + `${key}:${style[key]};`;
		}, '');
	};

	return {
		duration: params.duration ?? 200,
		delay: 0,
		css: (t) => {
			const y = scaleConversion(t, [0, 1], [params.y ?? 5, 0]);
			const x = scaleConversion(t, [0, 1], [params.x ?? 0, 0]);
			const scale = scaleConversion(t, [0, 1], [params.start ?? 0.95, 1]);

			return styleToString({
				transform: `${transform} translate3d(${x}px, ${y}px, 0) scale(${scale})`,
				opacity: t
			});
		},
		easing: cubicOut
	};
};

export function copyText(text: string) {
	return new Promise((resolve, reject) => {
		if (window.location.protocol === 'http:') {
			// 创建一个隐藏的textarea元素
			const textarea = document.createElement('textarea');
			textarea.value = text;
			document.body.appendChild(textarea);

			// 选中textarea中的文本
			textarea.select();
			textarea.setSelectionRange(0, 99999); // 对移动设备进行兼容处理

			try {
				// 复制文本到剪贴板
				const successful = document.execCommand('copy');
				const msg = successful ? 'successful' : 'unsuccessful';
				console.log('Copying text was ' + msg);
			} catch (err) {
				resolve(err);
			}

			// 移除textarea元素
			document.body.removeChild(textarea);
			resolve(true);
		} else {
			// 使用navigator.clipboard.writeText()方法
			navigator.clipboard.writeText(text).then(
				() => {
					resolve(true);
				},
				(err) => {
					reject(err);
				}
			);
		}
	});
}
