export default function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-danger-border bg-danger-bg px-4 py-3 text-sm text-danger">
      <span className="mt-[3px] h-1.5 w-1.5 flex-none rounded-full bg-danger" aria-hidden />
      <span>{message}</span>
    </div>
  );
}
