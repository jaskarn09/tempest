/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.html',
  ],
  theme: {
    extend: {
      colors: {
        'fire-red': '#dc2626',
        'safe-green': '#16a34a',
      }
    },
  },
  plugins: [],
}
