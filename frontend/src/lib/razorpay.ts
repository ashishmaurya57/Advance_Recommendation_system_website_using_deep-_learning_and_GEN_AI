import type { PaymentStart } from "./types";

type RazorpayResponse = {
  razorpay_order_id: string;
  razorpay_payment_id: string;
  razorpay_signature: string;
};

type RazorpayInstance = { open: () => void; on: (event: string, cb: () => void) => void };

declare global {
  interface Window {
    Razorpay?: new (options: Record<string, unknown>) => RazorpayInstance;
  }
}

let loader: Promise<void> | null = null;

function loadCheckout() {
  loader ??= new Promise<void>((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://checkout.razorpay.com/v1/checkout.js";
    s.onload = () => resolve();
    s.onerror = () => {
      loader = null;
      reject(new Error("Couldn't load the payment window. Check your connection."));
    };
    document.body.appendChild(s);
  });
  return loader;
}

/** Opens Razorpay Checkout. Resolves with the signed response, or null if the user closed it. */
export async function openCheckout(
  payment: PaymentStart,
  prefill: { name: string; email: string; contact: string },
): Promise<RazorpayResponse | null> {
  await loadCheckout();
  if (!window.Razorpay) throw new Error("Payment window is unavailable.");
  const Razorpay = window.Razorpay;
  return new Promise((resolve) => {
    const rzp = new Razorpay({
      key: payment.key_id,
      order_id: payment.order_id,
      amount: payment.amount,
      currency: payment.currency,
      name: "BookTown",
      description: payment.product.name,
      prefill,
      theme: { color: getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() },
      handler: (res: RazorpayResponse) => resolve(res),
      modal: { ondismiss: () => resolve(null) },
    });
    rzp.open();
  });
}
