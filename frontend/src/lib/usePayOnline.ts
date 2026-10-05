import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";

import { queries, useStartPayment, useVerifyPayment } from "./queries";
import { openCheckout } from "./razorpay";
import { useToast } from "./toast";

/** Start a Razorpay payment for one book, verify it on the server, then go to orders. */
export function usePayOnline() {
  const { data: user } = useQuery(queries.me());
  const start = useStartPayment();
  const verify = useVerifyPayment();
  const toast = useToast();
  const navigate = useNavigate();
  const [payingId, setPayingId] = useState<number | null>(null);

  async function pay(productId: number) {
    if (!user) return navigate({ to: "/signin", search: { redirect: location.pathname } });
    setPayingId(productId);
    try {
      const payment = await start.mutateAsync(productId);
      const result = await openCheckout(payment, { name: user.name, email: user.email, contact: user.mobile });
      if (!result) return; // closed the window
      const { message } = await verify.mutateAsync(result);
      toast(message);
      navigate({ to: "/orders" });
    } catch (e) {
      toast(e instanceof Error ? e.message : "Payment failed.", "error");
    } finally {
      setPayingId(null);
    }
  }

  return { pay, payingId };
}
