import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 16, children, ...rest }: IconProps & { children: React.ReactNode }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" {...rest}>
      {children}
    </svg>
  );
}

export const Check = (p: IconProps) => <Icon {...p}><path d="M20 6 9 17l-5-5" /></Icon>;
export const Cross = (p: IconProps) => <Icon {...p}><path d="M18 6 6 18M6 6l12 12" /></Icon>;
export const Minus = (p: IconProps) => <Icon {...p}><path d="M5 12h14" /></Icon>;
export const Dot = (p: IconProps) => <Icon {...p}><circle cx="12" cy="12" r="4" fill="currentColor" stroke="none" /></Icon>;
export const Ring = (p: IconProps) => <Icon {...p}><circle cx="12" cy="12" r="5" /></Icon>;
export const Alert = (p: IconProps) => <Icon {...p}><path d="M12 9v4M12 17h.01" /><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" /></Icon>;
export const Shield = (p: IconProps) => <Icon {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" /></Icon>;
export const Database = (p: IconProps) => <Icon {...p}><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5" /><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3" /></Icon>;
export const Sun = (p: IconProps) => <Icon {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Icon>;
export const Moon = (p: IconProps) => <Icon {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" /></Icon>;
export const ThumbUp = (p: IconProps) => <Icon {...p}><path d="M7 10v12M15 5.9 14 10h5.8a2 2 0 0 1 2 2.3l-1.4 8A2 2 0 0 1 18.4 22H7V10l4.6-7.7A1.6 1.6 0 0 1 14.6 3c.7.6 1 1.5.4 2.9Z" /></Icon>;
export const ThumbDown = (p: IconProps) => <Icon {...p}><path d="M17 14V2M9 18.1 10 14H4.2a2 2 0 0 1-2-2.3l1.4-8A2 2 0 0 1 5.6 2H17v12l-4.6 7.7a1.6 1.6 0 0 1-3-.7c-.1-.6.1-1.2.6-2.9Z" /></Icon>;
export const Play = (p: IconProps) => <Icon {...p}><path d="m6 4 14 8-14 8Z" /></Icon>;
export const History = (p: IconProps) => <Icon {...p}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5M12 7v5l3 2" /></Icon>;
export const Sort = ({ dir, ...p }: IconProps & { dir: "asc" | "desc" | null }) => (
  <Icon {...p}>{dir === "asc" ? <path d="m7 14 5-5 5 5" /> : dir === "desc" ? <path d="m7 10 5 5 5-5" /> : <path d="m8 9 4-4 4 4M8 15l4 4 4-4" />}</Icon>
);
