import pandas as pd
from transformers import pipeline
nlp = None


def load_data():
    file_path = "data/freelancers.csv"
    try:
        data = pd.read_csv(file_path)
        print(f"Загружено {len(data)} записей")
        print(f"Сумма Earnings_USD: {data['Earnings_USD'].sum()}")
        return data
    except FileNotFoundError:
        print("Файл не найден!")
        return None


def initialize_nlp():
    global nlp
    if nlp is None:
        print("Загрузка языковой модели...")
        nlp = pipeline("question-answering", model="deepset/roberta-base-squad2")
        print("Модель загружена!")
    return nlp


def compare_earnings_by_payment_method(data):
    earnings_by_payment = data.groupby("Payment_Method")["Earnings_USD"].mean()
    crypto_earnings = earnings_by_payment.get("Crypto", 0)
    mobile_banking_earnings = earnings_by_payment.get("Mobile Banking", 0)
    if crypto_earnings > mobile_banking_earnings:
        difference = crypto_earnings - mobile_banking_earnings
        return f"Фрилансеры, принимающие криптовалюту, зарабатывают на {difference:.2f} USD больше."
    else:
        difference = mobile_banking_earnings - crypto_earnings
        return f"Фрилансеры с мобильным банкингом зарабатывают на {difference:.2f} USD больше."


def earnings_by_region(data):
    earnings_by_region = data.groupby("Client_Region")["Earnings_USD"].mean()
    result = "Средний доход по регионам:\n"
    for region, earnings in earnings_by_region.items():
        result += f"{region}: {earnings:.2f} USD\n"
    return result


def experts_with_few_projects(data):
    experts = data[data["Experience_Level"] == "Expert"]
    if len(experts) == 0:
        return "В данных нет фрилансеров с уровнем Expert."
    few_projects = experts[experts["Job_Completed"] < 100]
    percentage = (len(few_projects) / len(experts)) * 100
    return f"{percentage:.2f}% экспертов выполнили менее 100 проектов."


def high_earners_asia(data):
    asia_freelancers = data[data["Client_Region"] == "Asia"]
    if len(asia_freelancers) == 0:
        return "В данных нет фрилансеров из Азии."
    high_earners = len(asia_freelancers[asia_freelancers["Earnings_USD"] > 2000])
    percentage = (high_earners / len(asia_freelancers)) * 100
    return f"Процент фрилансеров из Азии с доходом >2000 USD: {percentage:.2f}%."


def it_earnings(data):
    it_earnings = data[data["Job_Category"] == "App Development"]["Earnings_USD"].mean()
    other_earnings = data[data["Job_Category"] != "App Development"]["Earnings_USD"].mean()
    return f"Средний доход в IT: {it_earnings:.2f} USD, в других категориях: {other_earnings:.2f} USD."


def prepare_context(data):
    earnings_by_payment = data.groupby("Payment_Method")["Earnings_USD"].mean()
    earnings_by_region = data.groupby("Client_Region")["Earnings_USD"].mean()
    experts = data[data["Experience_Level"] == "Expert"]
    experts_percentage = (len(experts[experts["Job_Completed"] < 100]) / len(experts)) * 100 if len(experts) > 0 else 0
    it_earnings = data[data["Job_Category"] == "App Development"]["Earnings_USD"].mean() or 0
    other_earnings = data[data["Job_Category"] != "App Development"]["Earnings_USD"].mean() or 0
    asia_freelancers = data[data["Client_Region"] == "Asia"]
    asia_high_percentage = (len(asia_freelancers[asia_freelancers["Earnings_USD"] > 2000]) / len(asia_freelancers)) * 100 if len(asia_freelancers) > 0 else 0

    context = (
        f"Средний доход по способам оплаты: Криптовалюта - {earnings_by_payment.get('Crypto', 0):.2f} USD, "
        f"Мобильный банкинг - {earnings_by_payment.get('Mobile Banking', 0):.2f} USD. "
        f"Средний доход по регионам: {', '.join([f'{region}: {earnings:.2f} USD' for region, earnings in earnings_by_region.items()])}. "
        f"Процент экспертов с менее чем 100 проектами: {experts_percentage:.2f}%. "
        f"Средний доход в IT: {it_earnings:.2f} USD, в других категориях: {other_earnings:.2f} USD. "
        f"Процент фрилансеров из Азии с доходом >2000 USD: {asia_high_percentage:.2f}%."
    )
    return context


def post_process_answer(query, answer, context, data):
    if "насколько больше" in query.lower():
        earnings_by_payment = data.groupby("Payment_Method")["Earnings_USD"].mean()
        crypto = earnings_by_payment.get("Crypto", 0)
        mobile = earnings_by_payment.get("Mobile Banking", 0)
        paypal = earnings_by_payment.get("PayPal", 0)

        if "криптовалют" in query.lower():
            difference = crypto - mobile
            if difference > 0:
                return f"Фрилансеры с криптовалютой зарабатывают на {difference:.2f} USD больше."
            else:
                return f"Фрилансеры с мобильным банкингом зарабатывают на {abs(difference):.2f} USD больше."
        elif "мобильным банкингом" in query.lower() and "paypal" in query.lower():
            difference = mobile - paypal
            if difference > 0:
                return f"{difference:.2f} USD"
            else:
                return f"Фрилансеры с PayPal зарабатывают на {abs(difference):.2f} USD больше."

    elif "средний доход" in query.lower():
        earnings_by_region = data.groupby("Client_Region")["Earnings_USD"].mean()
        if "азии" in query.lower():
            return f"{earnings_by_region.get('Asia', 0):.2f} USD"
        elif "европе" in query.lower():
            return f"{earnings_by_region.get('Europe', 0):.2f} USD"
        elif "it" in query.lower():
            it_earnings = data[data["Job_Category"] == "App Development"]["Earnings_USD"].mean()
            other_earnings = data[data["Job_Category"] != "App Development"]["Earnings_USD"].mean()
            return f"Средний доход в IT: {it_earnings:.2f} USD, в других категориях: {other_earnings:.2f} USD"

    elif "процент" in query.lower():
        if "экспертов" in query.lower() and "менее 100" in query.lower():
            experts = data[data["Experience_Level"] == "Expert"]
            few_projects = experts[experts["Job_Completed"] < 100]
            percentage = (len(few_projects) / len(experts)) * 100 if len(experts) > 0 else 0
            return f"{percentage:.2f}%"
        elif "азии" in query.lower() and "больше 2000" in query.lower():
            asia_freelancers = data[data["Client_Region"] == "Asia"]
            high_earners = len(asia_freelancers[asia_freelancers["Earnings_USD"] > 2000])
            percentage = (high_earners / len(asia_freelancers)) * 100 if len(asia_freelancers) > 0 else 0
            return f"{percentage:.2f}%"
        elif "австралии" in query.lower() and "больше 5000" in query.lower():
            australia_experts = data[(data["Client_Region"] == "Australia") & (data["Experience_Level"] == "Expert")]
            high_earners = len(australia_experts[australia_experts["Earnings_USD"] > 5000])
            percentage = (high_earners / len(australia_experts)) * 100 if len(australia_experts) > 0 else 0
            return f"{percentage:.2f}%"

    return answer


def main():
    data = load_data()
    if data is None:
        return
    nlp = initialize_nlp()
    context = prepare_context(data)
    print("Контекст:", context)

    while True:
        print("\nВведите запрос (или 'exit' для выхода):")
        query = input("> ")
        if query.lower() == "exit":
            break
        try:
            result = nlp(question=query, context=context)
            final_answer = post_process_answer(query, result['answer'], context, data)
            print(f"Ответ: {final_answer}")
        except Exception as e:
            print(f"Ошибка обработки запроса: {e}")
            print("Попробуйте переформулировать запрос.")


if __name__ == "__main__":
    main()
