#include <iostream>
#include <string>
#include <map>

using namespace std;

class Account 
{
    protected:
        static long long int acc_num_gen;  
        float balance;
        long long int accountnumber;  

    public:
        Account() : balance(0), accountnumber(acc_num_gen++) {}

        Account(float bal) : balance(bal), accountnumber(acc_num_gen++)
        {
            if (balance < 0) balance = 0;
        }

        float Balance() const 
        {
            return balance;
        }

        long long int AccountNumber() const 
        { 
            return accountnumber; 
        }

        virtual bool Deposit(float amount)  
        {
            if (amount >= 0)
            {
                balance += amount;
                return true;
            }
            return false;
        }

        virtual float Withdraw(float amount) 
        {
            if(amount >= 0 && balance >= amount)
            {
                balance -= amount;
                return amount;
            }
            return 0;
        }

        virtual void Process() {}  

        friend ostream& operator<<(ostream& os, const Account& acc)
        {
            os << "Account #" << acc.accountnumber << " | Balance: $" << acc.balance;
            return os;
        }
};

long long int Account::acc_num_gen = 16210101000;  

class Checking : public Account
{
    private:
        float fee;

    public:
        Checking() : Account(), fee(0) {}
        Checking(float bal, float fee_val): Account(bal), fee(fee_val) {}

        bool Deposit(float amount) override  
        {
            return Account::Deposit(amount);
        }

        float Withdraw(float amount) override  
        {
            float withdrawn_amount = Account::Withdraw(amount);
            if (withdrawn_amount > 0 && balance < 500)
            {
                balance -= fee;
            }
            return withdrawn_amount;
        }

        void Process() override
        {
            if (balance < 500)
            {
                balance -= 10;  
            }
        }

        friend ostream& operator<<(ostream& os, const Checking& acc)
        {
            os << acc.AccountNumber() << " Checking | Balance: $" << acc.Balance() << " | Fee: $" << acc.fee;
            return os;
        }
};

class Savings : public Account
{
    private:
        float fee;

    public:
        Savings() : Account(), fee(0) {}
        Savings(float bal, float fee_val) : Account(bal), fee(fee_val) {}

        bool Deposit(float amount) override
        {
            return Account::Deposit(amount);
        }

        float Withdraw(float amount) override
        {
            float withdrawn_amount = Account::Withdraw(amount);
            if (withdrawn_amount > 0 && balance < 500)
            {
                balance -= fee;
            }
            return withdrawn_amount;
        }

        void Process() override
        {
            if (balance < 500)
            {
                balance -= 3.50;
                balance *= 1.004026;  
            }
        }

        friend ostream& operator<<(ostream& os, const Savings& acc)
        {
            os << acc.AccountNumber() << " Savings | Balance: $" << acc.Balance() << " | Fee: $" << acc.fee;
            return os;
        }
};


map<long long int, Account*> accounts;

void app()
{
    int option = 0;
    do
    {
        cout << "1. Start an account\n";
        cout << "2. Perform a transaction\n";
        cout << "0. Exit\n";
        cout << "Enter a number: ";
        cin >> option;

        if (option == 1)
        {
            int accType;
            float initialBalance;
            cout << "Enter account type (Checkings: 1, Savings: 2): ";
            cin >> accType;
            cout << "Enter initial balance: ";
            cin >> initialBalance;

            if (accType == 1)
            {
                Checking* acc = new Checking(initialBalance, 8.00);
                accounts[acc->AccountNumber()] = acc;
                cout << "Checking account created: " << *acc << endl;
            }
            else if (accType == 2)
            {
                Savings* acc = new Savings(initialBalance, 5.00);
                accounts[acc->AccountNumber()] = acc;
                cout << "Savings account created: " << *acc << endl;
            }
            else
            {
                cout << "Invalid account type" << endl;
            }
        }
        else if (option == 2)
        {
            long long int accNumber;
            int transType;
            float amount;
            cout << "Enter account #: ";
            cin >> accNumber;

            if (accounts.find(accNumber) == accounts.end())
            {
                cout << "Account not found.\n";
                continue;
            }

            Account* acc = accounts[accNumber];

            cout << "Enter transaction type (1: Deposit, 2: Withdraw): ";
            cin >> transType;
            cout << "Enter amount: ";
            cin >> amount;

            if (transType == 1)
            {
                if (acc->Deposit(amount))
                {
                    cout << "Deposit successful. New balance: " << acc->Balance() << endl;
                }
                else
                {
                    cout << "Deposit failed.\n";
                }
            }
            else if (transType == 2)
            {
                float withdrawn = acc->Withdraw(amount);
                if (withdrawn > 0)
                {
                    cout << "Withdrawal successful. Amount: " << withdrawn << " | New balance: " << acc->Balance() << endl;
                }
                else
                {
                    cout << "Withdrawal failed.\n";
                }
            }
        }
    } while (option != 0);

    cout << "Closing app.\n";
}

int main()
{
    app();
    return 0;
}
