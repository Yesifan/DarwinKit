// src/hooks.js
import { i18n } from '$lib/i18n';

export const reroute = ({ url }: { url: URL }) => {
	if (url.pathname.startsWith('/zh/docs/')) {
		const pathname = url.pathname.replace('/zh/docs', '/docs/zh');
		return pathname;
	} else {
		return i18n.reroute()({ url });
	}
};
