type ActionIconSource = { label?: string; kind?: string; ability?: string };

/** Keep the action's text accessible; these icons are supplemental visual cues. */
export function ActionIcons({ action }: { action: ActionIconSource }) {
  const description = `${action.label ?? ""} ${action.ability ?? ""}`.replaceAll("_", " ");
  const gains = (resource: string) => new RegExp(`\\bgain\\s+(?:\\d+\\s+)?${resource}\\b`, "i").test(description);
  const authority = gains("authority");
  const combat = gains("combat") || /^attack(?:_|$)/.test(action.kind ?? "") || /\battack(?:ing)?\b/i.test(description);
  const trade = gains("trade");
  return <>{authority ? <svg className="action-resource-icon action-resource-authority" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 3 6v6c0 5 9 10 9 10s9-5 9-10V6Z" fill="currentColor" /><path d="m8 12 3 3 5-6" fill="none" stroke="#fff" strokeWidth="2" /></svg> : null}{combat ? <svg className="action-resource-icon action-resource-combat" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3" /><path d="M12 0v5m0 14v5M0 12h5m14 0h5" /></svg> : null}{trade ? <svg className="action-resource-icon action-resource-trade" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="currentColor" /><circle cx="12" cy="12" r="7" fill="none" stroke="#785600" strokeWidth="1.5" /><path d="M12 7v10m-3-8h4.5a2 2 0 0 1 0 4h-3a2 2 0 0 0 0 4H15" fill="none" stroke="#785600" strokeWidth="1.5" /></svg> : null}</>;
}
