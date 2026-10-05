// Mirrors the FastAPI response models in backend/app/schemas.

export interface Category {
  id: number;
  name: string;
  image: string | null;
}

export interface BookCard {
  id: number;
  name: string;
  image: string | null;
  language: string;
  category: Category;
  price: number;
  mrp: number;
}

export interface Review {
  id: number;
  user_name: string;
  rating: number;
  comment: string;
  created_at: string;
}

export type Reaction = "like" | "dislike";

export interface BookDetail extends BookCard {
  description: string;
  publisher: string;
  hardcover: string;
  published: string;
  pdf_url: string | null;
  likes: number;
  dislikes: number;
  average_rating: number | null;
  reviews: Review[];
  my_reaction: Reaction | null;
}

export interface User {
  email: string;
  name: string;
  dob: string | null;
  mobile: string;
  address: string;
  avatar: string | null;
  interests: string[];
}

export interface CartItem {
  id: number;
  added: string;
  product: BookCard;
}

export interface Cart {
  items: CartItem[];
  total: number;
}

export interface Order {
  id: number;
  status: string;
  date: string;
  product: BookCard;
}

export interface SearchResult {
  query: string;
  products: BookCard[];
  message: string | null;
}

export interface Recommendations {
  products: BookCard[];
  computing: boolean;
}

export interface PaymentStart {
  key_id: string;
  order_id: string;
  amount: number;
  currency: string;
  product: BookCard;
}

export interface Message {
  message: string;
}
