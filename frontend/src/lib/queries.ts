import { queryOptions, useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "./api";
import type {
  BookCard,
  BookDetail,
  Cart,
  Category,
  Message,
  Order,
  PaymentStart,
  Reaction,
  Recommendations,
  Review,
  SearchResult,
  User,
} from "./types";

export const queries = {
  me: () =>
    queryOptions({ queryKey: ["me"], queryFn: () => api.get<User | null>("/auth/me"), staleTime: 60_000 }),
  home: () =>
    queryOptions({
      queryKey: ["home"],
      queryFn: () => api.get<{ categories: Category[]; latest: BookCard[] }>("/home"),
    }),
  categories: () =>
    queryOptions({ queryKey: ["categories"], queryFn: () => api.get<Category[]>("/categories") }),
  books: (category?: number) =>
    queryOptions({
      queryKey: ["books", category ?? "all"],
      queryFn: () =>
        api.get<{ category: Category | null; products: BookCard[] }>(
          category ? `/products?category=${category}` : "/products",
        ),
    }),
  book: (id: number) =>
    queryOptions({ queryKey: ["book", id], queryFn: () => api.get<BookDetail>(`/products/${id}`) }),
  search: (q: string) =>
    queryOptions({
      queryKey: ["search", q],
      queryFn: () => api.get<SearchResult>(`/search?q=${encodeURIComponent(q)}`),
      staleTime: 5 * 60_000,
    }),
  recommendations: () =>
    queryOptions({
      queryKey: ["recommendations"],
      queryFn: () => api.get<Recommendations>("/recommendations"),
      // Recommendations are computed in the background; poll quickly while that runs.
      refetchInterval: (q) => (q.state.data?.computing ? 4_000 : 30_000),
    }),
  cart: () => queryOptions({ queryKey: ["cart"], queryFn: () => api.get<Cart>("/cart") }),
  orders: () => queryOptions({ queryKey: ["orders"], queryFn: () => api.get<Order[]>("/orders") }),
  languages: () =>
    queryOptions({
      queryKey: ["reader-languages"],
      queryFn: () => api.get<string[]>("/reader/languages"),
      staleTime: Infinity,
    }),
};

export function useSignIn() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { email: string; password: string }) => api.post<User>("/auth/signin", body),
    onSuccess: (user) => {
      qc.setQueryData(queries.me().queryKey, user);
      qc.invalidateQueries({ predicate: (q) => q.queryKey[0] !== "me" });
    },
  });
}

export function useSignUp() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (form: FormData) => api.post<User>("/auth/signup", form),
    onSuccess: (user) => qc.setQueryData(queries.me().queryKey, user),
  });
}

export function useSignOut() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<Message>("/auth/signout"),
    onSuccess: () => {
      qc.setQueryData(queries.me().queryKey, null);
      qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" });
    },
  });
}

export function useUpdateProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (form: FormData) => api.put<User>("/profile", form),
    onSuccess: (user) => {
      qc.setQueryData(queries.me().queryKey, user);
      qc.invalidateQueries({ queryKey: ["recommendations"] });
    },
  });
}

export function useAddToCart() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (productId: number) => api.post<Message>("/cart", { product_id: productId }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cart"] }),
  });
}

export function useRemoveFromCart() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (itemId: number) => api.delete<Message>(`/cart/${itemId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cart"] }),
  });
}

export function usePlaceOrder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { product_id: number; from_cart?: boolean }) => api.post<Message>("/orders", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["orders"] });
      qc.invalidateQueries({ queryKey: ["cart"] });
    },
  });
}

export function useCancelOrder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (orderId: number) => api.delete<Message>(`/orders/${orderId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["orders"] }),
  });
}

export function useAddReview(bookId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { rating: number; comment: string }) =>
      api.post<Review>(`/products/${bookId}/reviews`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["book", bookId] }),
  });
}

export function useReact(bookId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (action: Reaction) =>
      api.post<{ likes: number; dislikes: number; my_reaction: Reaction | null }>(
        `/products/${bookId}/reaction`,
        { action },
      ),
    onSuccess: (data) =>
      qc.setQueryData(queries.book(bookId).queryKey, (old) => (old ? { ...old, ...data } : old)),
  });
}

export function useStartPayment() {
  return useMutation({
    mutationFn: (productId: number) => api.post<PaymentStart>("/payments/start", { product_id: productId }),
  });
}

export function useVerifyPayment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { razorpay_order_id: string; razorpay_payment_id: string; razorpay_signature: string }) =>
      api.post<Message>("/payments/verify", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["orders"] });
      qc.invalidateQueries({ queryKey: ["cart"] });
    },
  });
}

export function useContact() {
  return useMutation({
    mutationFn: (body: { name: string; email: string; mobile: string; message: string }) =>
      api.post<Message>("/contact", body),
  });
}
