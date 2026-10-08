# syntax=docker/dockerfile:1
# QueryGuard web UI: Vite builds the static app, nginx (unprivileged image,
# non-root, port 8080) serves it and proxies the API so the browser stays
# same-origin -- the same arrangement as the Vite dev proxy.

FROM node:24-slim AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
