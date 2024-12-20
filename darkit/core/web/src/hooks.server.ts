// src/hooks.server.js
import { i18n } from '$lib/i18n';
export const handle = ({ event, resolve }) => {
	return i18n.handle()({ event, resolve });
};
